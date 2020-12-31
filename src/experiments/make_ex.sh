fnum=$(ls -l | wc -l)
fnum=$(($fnum - 2))
fnum=${fnum//[[:blank:]]}
fsize=${#fnum}

if [ $fsize = 1 ]; then
    fnum=00$fnum
elif [ $fsize = 2 ]; then
    fnum=0$fnum
fi

fname=exp_$fnum

mkdir $fname
cd $fname

touch run.sh
touch main.py
touch config.yml
touch description.txt
touch .gitignore