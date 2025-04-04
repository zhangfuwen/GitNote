
# array
declare -a arr
arr=(1 2 4)
arr[2]=5

echo ${arr[@]} # print all elements
echo ${!arr[@]} # print all keys
echo ${#arr[@]} # print array length

for i in `seq 0 3`; do
  if [[ -z "${arr[$i]}" ]]; then 
    echo "arr["$i"] not set"; 
  else
    echo "arr["$i"]" ${arr[$i]}
  fi
done

# add new elements
arr+=(7 9)
# delete elements, causes a hole
unset arr[3]

for i in ${!arr[@]}; do
  echo "arr[$i]:${arr[$i]}"
done

# dict
declare -A hm
hm=( 
  [index_foo]=value_foo 
  [index_bar]=value_bar 
  [index_xyz]=value_xyz 
)
hm[hello]=1
hm[world]=1
hm[hello]=2
if [ "${hm[hello]}" ] ; then echo ${hm[hello]} ; fi
echo ${!hm[@]} # print all keys
echo ${hm[@]} # print all elements
echo ${#hm[@]} # print array length
