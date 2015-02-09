for i in $(find . -name "*.png"); do convert $i -strip $i; done
