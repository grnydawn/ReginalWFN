#!/bin/bash

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--master)
            echo "Option master is set with argument '$2'"
            shift
            #exit 0
            ;;
        -v|--virtualenv)
            echo "Option virtuanenv is set with argument '$2'"
            shift
            #exit 0
            ;;
        *)
            echo "Unknown option: $1"
            #exit 1
            ;;
    esac
    shift
done

#ARGS=$(getopt -o m:v: -l master:,virtualenv: -- "$@")
#if [ $? -ne 0 ]; then
#    echo "INFO: getopt is not available." >&2
#fi
#
#eval set -- "$ARGS"
#
#while true; do
#    case "$1" in
#        -m|--master)
#            echo "Option master is set with argument '$2'"
#            shift 2
#            ;;
#        -v|--virtualenv)
#            echo "Option virtuanenv is set with argument '$2'"
#            shift 2
#            ;;
#        --)
#            shift
#            break
#            ;;
#        *)
#            break
#            ;;
#    esac
#done
#
echo "Remaining arguments:"
for arg in "$@"; do
    echo "-> $arg"
done
