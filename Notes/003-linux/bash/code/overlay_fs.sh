mkdir 1 2 3 4 5 6 7
for i in 1 2 3 4 5 6 7; do
  touch $i/$i.txt
done

mount -t overlay overlay -o lowerdir=1:2:3:4:5 7
ls 7