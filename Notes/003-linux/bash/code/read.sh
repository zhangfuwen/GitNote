echo "read a line:"
read -r line
echo "line:$line"

echo "read a var"
read var
echo "var:$var"

echo "read two vars"
read var1 var2
echo "var1:$var1 var2:$var2"

read -p 'Username: ' uservar
read -sp 'Password: ' passvar
echo
echo Thankyou $uservar we now have your login details

# ~/bashexamples$ bash read.sh
# read a line:
# asdf asdfa
# line:asdf asdfa
# read a var
# asfd asdfa
# var:asfd asdfa
# read two vars
# asdf asdfa
# var1:asdf var2:asdfa
# Username: dean
# Password: 
# Thankyou dean we now have your login details