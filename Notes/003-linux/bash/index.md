---
title: bash
---
# data_structure

[examples](./code/data_structure.sh)

## array

```bash
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

```

## map

```bash
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
```

# zenity

## select from a list

```bash
zenity --list --title="select" --column "selection" "A" "B X" C D
```

select a disk

```bash
IFS=$'\n' disks=$(zenity --list --title="select" --column "selection" $(lsblk -no path,vendor,size,type /dev/sd*  2> /dev/null| grep disk))
if [[ $? == 0 ]]; do
    selected_disk=$(echo $disks | awk '{print $1;}')
    echo $selected_disk
fi

```

## input

input an url

```bash
url=$(zenity --title="win" --entry --text="input url")
if [[ $? != 0 ]]; then
    echo "nothing"
else
    echo $url
fi
```

## dd image from http url

```bash

command -v zenity || sudo apt -y install zenity
function select_disk() {
    IFS=$'\n' disks=$(zenity --list --title="select" --column "selection" $(lsblk -no path,vendor,size,type /dev/sd*  2> /dev/null| grep disk))
    if [[ $? == 0 ]]; then
        selected_disk=$(echo $disks | awk '{print $1;}')
        echo $selected_disk
        return 0
    fi
    return -1
}

function input_url() {
    url=$(zenity --title="win" --entry --text="input url")
    if [[ $? != 0 ]]; then
        echo "nothing"
        return -1
    else
        echo $url
        return 0
    fi
}

disk=$(select_disk) && url=$(input_url) && echo "dd if=$url of=$disk" \
&& curl -u zhangfuwen:zhangfuwen --silent $url | \
sudo dd conv=noerror,sync iflag=fullblock oflag=direct,sync status=progress bs=1M of=$disk

```

# wget显示进度

```bash
    wget --progress=bar:force "http://base.url.here/filename.txt" -O/your/destination/and/filename 2>&1 | zenity --title="File transfer in progress!" --progress --auto-close --auto-kill
```


# wget via ssh
```bash
ssh -C user@hostB "wget -O- http://website-C" >> file-from-website-C
```

# dd

```bash
dd if=.. of=.. status=progress oflag=direct,sync bs=10M
```

# 所有代码

{% for file in site.static_files %}
  {% if file.path contains "bash/code" -%}
     * [{{ file.path }}]({{ site.baseurl }}{{ file.path }})
  {%- endif %}
{% endfor %}