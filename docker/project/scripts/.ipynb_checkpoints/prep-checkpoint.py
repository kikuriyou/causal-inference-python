import os
from time import time
import gc
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

def load_data(project_id, pkl_file, term, is_rank=False):
    query1 = get_query1(term)
    query2 = get_query2_rank() if is_rank else get_query2()
    query3 = get_query3()
    query4 = get_query4_rank() if is_rank else get_query4()
    query = query1 + query2 + query3 + query4
    if os.path.exists(pkl_file):
        df = pd.read_pickle(pkl_file)
    else:
        df = pd.read_gbq(query=query, project_id=project_id, dialect='standard')
        df.to_pickle(pkl_file)
    return df
    
def get_query1(term):
    return f"""
    CREATE TEMP FUNCTION TREAT_START() AS (DATE('{term[0]}'));
    CREATE TEMP FUNCTION TREAT_END() AS (DATE('{term[1]}'));

    WITH
    raw_data AS (
    SELECT
        date,
        installTime,
      appUserId,
        age,
        osType,
        country,
        nintendoAccountId
    FROM
      GAME_SUMMARY_DAILY_JST.ACCESS
    WHERE
    --  _PARTITIONTIME BETWEEN TIMESTAMP('2019-10-19') AND TIMESTAMP('2019-11-18')
    --  _PARTITIONTIME BETWEEN TIMESTAMP(DATE_SUB(TREAT_START(), INTERVAL 30 DAY)) 
    -- 					AND TIMESTAMP(DATE_SUB(TREAT_START(), INTERVAL 1 DAY))
    --  _PARTITIONTIME BETWEEN TIMESTAMP('2019-11-19') AND TIMESTAMP('2019-11-25')
      _PARTITIONTIME BETWEEN TIMESTAMP(TREAT_START()) AND TIMESTAMP(TREAT_END())
      AND installTime <= TIMESTAMP(DATE_SUB(TREAT_START(), INTERVAL 31 DAY))
      AND isSubAccountLastUse IN (0, 2)
      AND isInvalidAccountByIpHash IN (0, 2)
    )

    , na_user AS (
    SELECT
        appUserId,
        MAX(nintendoAccountId) AS nintendoAccountId
    FROM
        raw_data
    GROUP BY
        1
    )

    -- OS, country, NA連携 はとりあえずmaxで絞る
    , active_users AS (
    SELECT
        a.appUserId,
        MIN(a.installTime) AS install_time,
        MAX(age) AS age,
        MAX(CASE WHEN osType = 'iOS' THEN 1 ELSE 0 END) AS is_ios,
        MAX(CASE WHEN country = 'JP' THEN 1 ELSE 0 END) AS is_jp,
        MAX(CASE WHEN b.nintendoAccountId IS NOT NULL THEN 1 ELSE 0 END) AS is_na
    FROM 
        raw_data a
    LEFT JOIN 
        na_user b
        ON a.appUserId = b.appUserId
    GROUP BY 1
    )

    , transaction_agg AS (
    SELECT
        appUserId,
        SUM(DISTINCT CASE WHEN type = 'purchase' THEN amount END) AS purchase_amt,
        COUNT(DISTINCT CASE WHEN type = 'purchase' THEN date END) AS purchase_dates,
        SUM(DISTINCT CASE WHEN type = 'spend_paid' THEN amount END) AS spend_paid_amt,
        COUNT(DISTINCT CASE WHEN type = 'spend_paid' THEN date END) AS spend_paid_dates,
        SUM(DISTINCT CASE WHEN type = 'spend_free' THEN vcAmount END) AS spend_free_amt,
        COUNT(DISTINCT CASE WHEN type = 'spend_free' THEN date END) AS spend_free_dates
    FROM
        GAME_SUMMARY_DAILY_JST.TRANSACTION
    WHERE
    --  _PARTITIONTIME BETWEEN TIMESTAMP('2019-10-19') AND TIMESTAMP('2019-11-18')
      _PARTITIONTIME BETWEEN TIMESTAMP(DATE_SUB(TREAT_START(), INTERVAL 30 DAY)) 
                        AND TIMESTAMP(DATE_SUB(TREAT_START(), INTERVAL 1 DAY))
    GROUP BY
        1
    )

    , session_agg AS (
    SELECT DISTINCT
        appUserId,
        SUM(sessionCount) OVER(PARTITION BY appUserId) AS sess_cnt,
        SUM(totalSessionTime) OVER(PARTITION BY appUserId) AS sess_time_total,
        MIN(CASE WHEN totalSessionTime > 0 THEN totalSessionTime END) OVER(PARTITION BY appUserId) AS sess_time_min,
        AVG(totalSessionTime) OVER(PARTITION BY appUserId) AS sess_time_mean,
        PERCENTILE_CONT(totalSessionTime, 0.5) OVER(PARTITION BY appUserId) AS sess_time_median, 
        MAX(totalSessionTime) OVER(PARTITION BY appUserId) AS sess_time_max
    FROM
        GAME_SUMMARY_DAILY_JST.SESSION
    WHERE
    --  _PARTITIONTIME BETWEEN TIMESTAMP('2019-10-19') AND TIMESTAMP('2019-11-18')
      _PARTITIONTIME BETWEEN TIMESTAMP(DATE_SUB(TREAT_START(), INTERVAL 30 DAY)) 
                        AND TIMESTAMP(DATE_SUB(TREAT_START(), INTERVAL 1 DAY))
    )

    -- ガチャ初日終了時点でのVC,キャラ所持量
    , possession AS (
    SELECT
        a.appUserId,
        IFNULL(a.vc_possession, 0) AS vc_possession,
        IFNULL(b.unit_possession, 0) AS unit_possession
    FROM
        (SELECT appUserId, SUM(vcurAmtFree + vcurAmtPaid) AS vc_possession
        FROM `NPFSNAPSHOT.VCBALANCE_*`
    -- 	WHERE SUBSTR(_TABLE_SUFFIX, -8) = '20191118'
        WHERE SUBSTR(_TABLE_SUFFIX, -8) = REPLACE(CAST(DATE_SUB(TREAT_START(), INTERVAL 1 DAY) AS STRING), '-', '')    -- '20191118'
        GROUP BY 1) a
    FULL JOIN
        (SELECT appUserId, COUNT(DISTINCT person) AS unit_possession
        FROM GAME_SUMMARY_DAILY_JST.HISTORY_DECK a
        LEFT JOIN GAME_ADHOC.CHARACTER b
            ON REPLACE(a.person, 'PID_', '') = b.name
    -- 	WHERE _PARTITIONTIME = TIMESTAMP('2019-11-18')
        WHERE _PARTITIONTIME = TIMESTAMP(DATE_SUB(TREAT_START(), INTERVAL 1 DAY))
        GROUP BY 1) b
        ON a.appUserId = b.appUserId
    )

    -- 直近3タームでのプレイ状況
    , prev_term AS (
    SELECT
        appUserId,
        MAX(CASE WHEN term_key = 'sc_0050' THEN cnt END) AS cnt_sc_0050,
        MAX(CASE WHEN term_key = 'sc_0051' THEN cnt END) AS cnt_sc_0051,
        MAX(CASE WHEN term_key = 'sc_0052' THEN cnt END) AS cnt_sc_0052,
        MAX(CASE WHEN term_key = 'sc_0053' THEN cnt END) AS cnt_sc_0053,
        MAX(CASE WHEN term_key = 'sc_0050' AND cnt > 0 THEN 1 END) AS is_played_sc_0050,
        MAX(CASE WHEN term_key = 'sc_0051' AND cnt > 0 THEN 1 END) AS is_played_sc_0051,
        MAX(CASE WHEN term_key = 'sc_0052' AND cnt > 0 THEN 1 END) AS is_played_sc_0052,
        MAX(CASE WHEN term_key = 'sc_0053' AND cnt > 0 THEN 1 END) AS is_played_sc_0053
    FROM (
        SELECT appUserId, JSON_EXTRACT_SCALAR(payload, '$.termKey') AS term_key, COUNT(*) AS cnt
        FROM EVENTS_DAILY_JST_CLUSTERED.skyCastle_v01 a
        JOIN GAME_MASTER.SKY_CASTLE b
            ON JSON_EXTRACT_SCALAR(a.payload, '$.termKey') = b.id
        WHERE 
    -- 		date BETWEEN '2019-10-22' AND '2019-11-18'
            date BETWEEN DATE_SUB(TREAT_START(), INTERVAL 28 DAY) AND DATE_SUB(TREAT_START(), INTERVAL 1 DAY)
            AND eventId = 'battleScore'
            AND FORMAT_TIMESTAMP('%F', openDate, 'Asia/Tokyo') 
                BETWEEN CAST(DATE_SUB(TREAT_START(), INTERVAL 28 DAY) AS STRING) 
                    AND CAST(DATE_SUB(TREAT_START(), INTERVAL 1 DAY) AS STRING)
        GROUP BY 1, 2)
    GROUP BY
        1
    )
    
    -- 直近4タームの最終グレード
    , daily_grade AS (
    SELECT DISTINCT
        date,
        appUserId,
        lastGrade + 1 AS lastGrade
    FROM
        GAME_SUMMARY_DAILY_JST.SKY_CASTLE
    WHERE
        _PARTITIONTIME BETWEEN TIMESTAMP(DATE_SUB(TREAT_START(), INTERVAL 28 DAY)) 
                            AND TIMESTAMP(DATE_SUB(TREAT_START(), INTERVAL 1 DAY))
    )

    , get_grade AS (
    SELECT
        --a.appUserId, lastGrade AS prev_grade
        a.appUserId, GREATEST(lastGrade, 10) AS prev_grade
        -- プレイ有無ではなく飛空城へのモチベーションを見たいため、グレード10以下は実質同等(放置)なのでまとめる
    FROM
        daily_grade a
    JOIN 
        (SELECT appUserId, MAX(date) AS date FROM daily_grade GROUP BY 1) b
        ON a.appUserId = b.appUserId AND a.date = b.date
    )

    -- 対象ターム終了翌日~次の伝承ガチャ開始までの継続状況
    -- 期間中の最終アクセスの翌日をevent_timeとする場合
    , churn_last AS (
    SELECT
        appUserId, 
        GREATEST(LEAST(
            DATE_DIFF(DATE_ADD(CAST(last_date AS DATE), INTERVAL 1 DAY), TREAT_END(), DAY), 
            30), 0) AS event_time_last
    FROM
        (SELECT DISTINCT appUserId, MAX(date) AS last_date
        FROM GAME_SUMMARY_DAILY_JST.ACCESS
    -- 	WHERE _PARTITIONTIME BETWEEN TIMESTAMP('2019-11-19') AND TIMESTAMP('2019-12-24')
        WHERE _PARTITIONTIME BETWEEN TIMESTAMP(TREAT_START()) 
                                AND TIMESTAMP(DATE_ADD(TREAT_END(), INTERVAL 30 DAY))
        GROUP BY 1)
    WHERE
        last_date IS NOT NULL
    )

    -- 社内定義(7日間アクセスなし)で最初に離脱した日をevent_timeとする場合
    , no_access AS (
    SELECT
        uniq.appUserId, 
        uniq.date,
        LAG(uniq.date, 6) OVER (PARTITION BY uniq.appUserId ORDER BY uniq.date ASC) AS date_6lag,
        DATE_SUB(CAST(uniq.date AS DATE), INTERVAL 6 DAY) AS date_6ago
    FROM (
        SELECT appUserId, date
        FROM
            (SELECT DISTINCT appUserId
            FROM GAME_SUMMARY_DAILY_JST.ACCESS
            WHERE _PARTITIONTIME BETWEEN TIMESTAMP(TREAT_START()) 
                                    AND TIMESTAMP(DATE_ADD(TREAT_END(), INTERVAL 30 DAY))) uniq_users
        CROSS JOIN 
            (SELECT DISTINCT date
            FROM GAME_SUMMARY_DAILY_JST.ACCESS
            WHERE _PARTITIONTIME BETWEEN TIMESTAMP(TREAT_START()) 
                                    AND TIMESTAMP(DATE_ADD(TREAT_END(), INTERVAL 30 DAY))) uniq_dates
        ) uniq
    LEFT JOIN
        (SELECT DISTINCT appUserId, date
        FROM GAME_SUMMARY_DAILY_JST.ACCESS
        WHERE _PARTITIONTIME BETWEEN TIMESTAMP(TREAT_START()) 
                                AND TIMESTAMP(DATE_ADD(TREAT_END(), INTERVAL 30 DAY))) access
        ON uniq.appUserId = access.appUserId AND uniq.date = access.date
    WHERE
        access.appUserId IS NULL
    )

    -- 継続ユーザーはここに含まれない
    , churn AS (
    SELECT
        appUserId,
        GREATEST(LEAST(
            DATE_DIFF(CAST(event_date AS DATE), TREAT_END(), DAY), 
            30), 0) AS event_time
    FROM (
        SELECT appUserId, MIN(date) AS event_date    -- 最初の離脱日
        FROM no_access
        WHERE date_6lag = CAST(date_6ago AS STRING)
        GROUP BY 1)
    )

    , covariates AS (
    SELECT
        a.appUserId,
    -- 	DATE_DIFF(CAST('2019-11-16' AS DATE), CAST(a.install_time AS DATE), DAY) AS elapsed_date,
        DATE_DIFF(DATE_SUB(TREAT_START(), INTERVAL 3 DAY), CAST(a.install_time AS DATE), DAY) AS elapsed_date,
        a.is_ios, a.is_jp, a.is_na, 
        IFNULL(b.purchase_amt, 0) AS purchase_amt, 
        IFNULL(b.purchase_dates, 0) AS purchase_dates, 
        IFNULL(b.spend_paid_amt, 0) AS spend_paid_amt, 
        IFNULL(b.spend_paid_dates, 0) AS spend_paid_dates, 
        IFNULL(b.spend_free_amt, 0) AS spend_free_amt, 
        IFNULL(b.spend_free_dates, 0) AS spend_free_dates,
        c.sess_cnt, 
        c.sess_time_total, 
        IFNULL(c.sess_time_min, 0) AS sess_time_min, 
        c.sess_time_mean, 
        c.sess_time_median, 
        c.sess_time_max,
        IFNULL(d.vc_possession, 0) AS vc_possession,
        IFNULL(d.unit_possession, 0) AS unit_possession,
        IFNULL(e.cnt_sc_0050, 0) AS cnt_sc_0050,
        IFNULL(e.cnt_sc_0051, 0) AS cnt_sc_0051,
        IFNULL(e.cnt_sc_0052, 0) AS cnt_sc_0052,
        IFNULL(e.cnt_sc_0053, 0) AS cnt_sc_0053,
        IFNULL(e.is_played_sc_0050, 0) AS is_played_sc_0050,
        IFNULL(e.is_played_sc_0051, 0) AS is_played_sc_0051,
        IFNULL(e.is_played_sc_0052, 0) AS is_played_sc_0052,
        IFNULL(e.is_played_sc_0053, 0) AS is_played_sc_0053,
        IFNULL(f.prev_grade, 10) AS prev_grade,
    """


def get_query2():
    return """
    
        PERCENT_RANK() OVER (ORDER BY IFNULL(b.purchase_amt, 0)) AS purchase_amt_rank,
        PERCENT_RANK() OVER (ORDER BY IFNULL(b.purchase_dates, 0)) AS purchase_dates_rank,
        PERCENT_RANK() OVER (ORDER BY IFNULL(b.spend_paid_amt, 0)) AS spend_paid_amt_rank,
        PERCENT_RANK() OVER (ORDER BY IFNULL(b.spend_paid_dates, 0)) AS spend_paid_dates_rank,
        PERCENT_RANK() OVER (ORDER BY IFNULL(b.spend_free_amt, 0)) AS spend_free_amt_rank,
        PERCENT_RANK() OVER (ORDER BY IFNULL(b.spend_free_dates, 0)) AS spend_free_dates_rank,

        PERCENT_RANK() OVER (ORDER BY d.vc_possession) AS vc_possession_rank,
        PERCENT_RANK() OVER (ORDER BY d.unit_possession) AS unit_possession_rank
    """

def get_query2_rank():
    return """
    
        PERCENT_RANK() OVER (ORDER BY sess_cnt) AS sess_cnt_rank,
        PERCENT_RANK() OVER (ORDER BY sess_time_total) AS sess_time_total_rank,
        PERCENT_RANK() OVER (ORDER BY IFNULL(c.sess_time_min, 0)) AS sess_time_min_rank,
        PERCENT_RANK() OVER (ORDER BY sess_time_mean) AS sess_time_mean_rank,
        PERCENT_RANK() OVER (ORDER BY sess_time_median) AS sess_time_median_rank,
        PERCENT_RANK() OVER (ORDER BY sess_time_max) AS sess_time_max_rank,

        PERCENT_RANK() OVER (ORDER BY IFNULL(e.cnt_sc_0050, 0)) AS cnt_sc_0050_rank,
        PERCENT_RANK() OVER (ORDER BY IFNULL(e.cnt_sc_0051, 0)) AS cnt_sc_0051_rank,
        PERCENT_RANK() OVER (ORDER BY IFNULL(e.cnt_sc_0052, 0)) AS cnt_sc_0052_rank,
        PERCENT_RANK() OVER (ORDER BY IFNULL(e.cnt_sc_0053, 0)) AS cnt_sc_0053_rank,
        PERCENT_RANK() OVER (ORDER BY IFNULL(f.prev_grade, 0)) AS prev_grade_rank
    """

def get_query3():
    return """
    
    FROM
        active_users a
    LEFT JOIN
        transaction_agg b
        ON a.appUserId = b.appUserId
    LEFT JOIN
        session_agg c
        ON a.appUserId = c.appUserId
    LEFT JOIN
        possession d
        ON a.appUserId = d.appUserId
    LEFT JOIN
        prev_term e
        ON a.appUserId = e.appUserId
    LEFT JOIN
        get_grade f
        ON a.appUserId = f.appUserId
    )

    , treatment AS (
    SELECT
        appUserId
    FROM (
        SELECT appUserId, COUNT(*) AS cnt
        FROM EVENTS_DAILY_JST_CLUSTERED.skyCastle_v01
        WHERE date BETWEEN TREAT_START() AND TREAT_END() AND eventId = 'battleScore'
        GROUP BY 1)
    WHERE
        cnt >= 1
    )

    -- 対象ターム終了翌日~次の伝承ガチャ開始までの課金/継続状況
    , outcome AS (
    SELECT
        a.appUserId, 
        a.pay_amt,
        CASE WHEN a.pay_amt > 0 THEN 1 ELSE 0 END AS is_pay,
        b.event_time_last,
        c.event_time
    FROM
        (SELECT appUserId, SUM(CASE WHEN type = 'purchase' THEN amount ELSE 0 END) AS pay_amt
        FROM GAME_SUMMARY_DAILY_JST.TRANSACTION
    -- 	WHERE _PARTITIONTIME BETWEEN TIMESTAMP('2019-11-26') AND TIMESTAMP('2019-12-24')
        WHERE _PARTITIONTIME BETWEEN TIMESTAMP(DATE_ADD(TREAT_END(), INTERVAL 1 DAY)) 
                                AND TIMESTAMP(DATE_ADD(TREAT_END(), INTERVAL 30 DAY))
            AND amount >= 0    -- 返金は一旦無視
        GROUP BY 1) a
    LEFT JOIN
        churn_last b
        ON a.appUserId = b.appUserId
    LEFT JOIN
        churn c
        ON a.appUserId = c.appUserId
    )
    
    , summarize AS (
    SELECT
        a.*,
        CASE WHEN b.appUserId IS NOT NULL THEN 1 ELSE 0 END AS is_play,
        IFNULL(c.pay_amt, 0) AS pay_amt,
        IFNULL(c.is_pay, 0) AS is_pay,
        IFNULL(c.event_time_last, 30) AS event_time_last,
        IFNULL(c.event_time, 30) AS event_time
    FROM
        covariates a
    LEFT JOIN
        treatment b
        ON a.appUserId = b.appUserId
    LEFT JOIN
        outcome c
        ON a.appUserId = c.appUserId
    )
    """

def get_query4():
    return """
    
    SELECT
        *
    FROM
        summarize
    """

def get_query4_rank():
    return """
    
    SELECT
        appUserId,
        sess_cnt_rank,
        sess_time_total_rank,
        sess_time_min_rank,
        sess_time_mean_rank,
        sess_time_median_rank,
        sess_time_max_rank,

        cnt_sc_0050_rank,
        cnt_sc_0051_rank,
        cnt_sc_0052_rank,
        cnt_sc_0053_rank,
        prev_grade_rank
    FROM
        summarize
    """