str="abcdABCDabcdABCD"

# remove prefix
echo ${str#a*d}  # ABCDabcdABCD
echo ${str##a*d} # ABCD

# remove suffix
echo ${str%ABCD}
echo ${str%%ABCD}

# replace
echo ${str/ABCD/1234}
echo ${str//ABCD/1234}

# replace front or end
echo ${str/#abcd/1234}
echo ${str/#ABCD/1234} # nothing happens
echo ${str/%ABCD/1234}
echo ${str/%abcd/1234} # nothing happens

# strlen
echo ${#str}

# match
pos=`expr index $str "b.*d"` # where
length=`expr match ${str:$pos-1} "b.*d"` # how long
echo $pos $length ${str:$pos-1:$length}
echo `expr match $str '.*\(b.*d\)'`  # 显示的是括号里的内容

# ~/bashexamples$ bash string_manipulation.sh 
# ABCDabcdABCD
# ABCD
# abcdABCDabcd
# abcdABCDabcd
# abcd1234abcdABCD
# abcd1234abcd1234
# 1234ABCDabcdABCD
# abcdABCDabcdABCD
# abcdABCDabcd1234
# abcdABCDabcdABCD
# 16
# 2 11 bcdABCDabcd
# bcd