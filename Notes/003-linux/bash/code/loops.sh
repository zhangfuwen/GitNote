for i in `seq 2 9`; do
  echo $i
done

files=`find . -name "*.sh"`
for file in $files; do
  echo $file
done

index=0
while true; do
  sleep 1
  let "index = $index + 1"
  echo $index
  if [[ $index -gt 10 ]]; then
    break;
  fi
done