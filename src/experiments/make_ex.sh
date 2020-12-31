dnum=$(ls -l | wc -l)
dnum=$(($fnum - 3))
dnum=${fnum//[[:blank:]]}
dsize=${#fnum}

if [ $fsize = 1 ]; then
    dnum=00$fnum
elif [ $fsize = 2 ]; then
    dnum=0$fnum
fi

dname=exp_$fnum
cp -r _template ${dname}