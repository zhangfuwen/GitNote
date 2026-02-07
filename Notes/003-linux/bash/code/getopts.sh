has_e=0
has_g=0
e_val=""

while getopts ":abcde:fg" Option
# e后面有一个冒号，代表e后面是要跟参数的
do
  case $Option in
    a ) echo a;;
    b ) echo b;;
    e)  echo e; has_e=1; e_val=$OPTARG; echo $OPTARG;;
    g ) echo g; has_g=1;;
  esac
done
shift $(($OPTIND - 1))
# 处理一下个

# All this is not nearly as complicated as it looks <grin>.
echo "has_e $has_e, has_g $has_g, e_val $e_val"


#~/LowNuttyChords$ bash getopts.sh -g -e hello
#g
#e
#hello
#has_e 1, has_g 1, e_val hello