#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/workspace/src/wrs_algorithm21"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/workspace/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/workspace/install/lib/python3/dist-packages:/workspace/build/wrs_algorithm/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/workspace/build/wrs_algorithm" \
    "/home/developer/.pyenv/versions/catkin_py3/bin/python3" \
    "/workspace/src/wrs_algorithm21/setup.py" \
    egg_info --egg-base /workspace/build/wrs_algorithm \
    build --build-base "/workspace/build/wrs_algorithm" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/workspace/install" --install-scripts="/workspace/install/bin"
