# /bin/bash
usage () {
    echo "usage: ./run_jupyter.sh PORT[port-number]"
}
# Command line argument
COMMAND=`basename $0`
if [ $# -ne 1 ]; then
    usage
    exit 1;
fi
echo "Command line argument: $1"
PORT=$1

jupyter lab --port=$PORT --ip=0.0.0.0 --allow-root --no-browser
