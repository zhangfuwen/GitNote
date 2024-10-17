File type tests
These options test for particular types of files. All cases will only return True (0) if the file exists.

 -b file      True if file is a Block special device. [[ -b demofile ]]

 -c file      True if file is a Character special device. [[ -c demofile ]]

 -d file      True if file is a Directory. [[ -d demofile ]]

 -e file      True if file Exists. [[ -e demofile ]]

 -f file      True if file is a regular File. [[ -f demofile ]]

 -g file      True if file has its set-group-id bit set. [[ -g demofile ]]

 -G file      True if file is owned by the current effective group id. [[ -G demofile ]]

 -k file      True if file has its "sticky" bit set. [[ -k demofile ]]

 -h file      True if file is a symbolic Link. [[ -h demofile ]]
 -L file      True if file is a symbolic Link. [[ -L demofile ]]

 -O file      True if file is owned by the current effective user id. [[ -O demofile ]]

 -p file      True if file is a named Pipe. [[ -p demofile ]]

 -r file      True if file is readable. [[ -r demofile ]]

 -S file      True if file is a Socket. [[ -S demofile ]]

 -s file      True if file has a Size greater than zero. [[ -s demofile ]]

 -t [FD]      True if FD is opened on a terminal.  If FD is omitted, it defaults to 1 (standard output). [[ -t demofile ]]

 -u file      True if file has its set-user-id bit set. [[ -L demofile ]]

 -w file      True if file is writable. [[ -w demofile ]]

 -x file      True if file is executable. [[ -x demofile ]]

file1 -ef file2    True if file1 and file2 have the same device and inode numbers,  i.e. they are hard links to each other.
File Age Tests (modification date)
file1 -nt file2     True if file1 is newer than file2. [[ demofile1 -nt $DEMO ]]

file1 -ot file2     True if file1 is older than file2. [[ $DEMO -ot demofile2 ]]
String tests
Comparison strings for test do not need to be quoted, though you can quote them to protect characters with special meaning to the shell, e.g. spaces.

Comparisons using New test [[ perform pattern matching against the string on the right hand side unless you quote the "string" on the right. This prevents any characters with special meaning in pattern matching from taking effect.

-z String      True if the length of String is zero.

-n String      True if the length of String is nonzero.
   String      True if the length of String is nonzero.

String1 = String2     True if the strings are equal.

[[ String1 = "String2" ]]  True if the strings are equal (Literal, no pattern match).

String1 != String2    True if the strings are not equal.
[[ a != b ]] && echo "a is not equal to b"

[[ a > b ]] || echo "a does not come after b"
[[ az < za ]] && echo "az comes before za"
[[ a = a ]] && echo "a equals a"

As of bash 4.1 (2010), string comparisons made with [[ and using < or > will respect the current locale.

Wildcard matching with [[
[[ $NAME = demo* ]] || echo "NAME does not start with 'demo': $name"
Numeric tests
These are normally used in conjunction with another non-math test operator somewhere in the expression.
For a purely numeric comparison, it is better to use (( )) instead of test or New test [[

The arguments must be entirely numeric (possibly negative), or the special expression -l STRING which evaluates to the length of STRING. The < and > operators can also be used with new test [[

ARG1 -eq ARG2   Returns true if ARG1 is equal to ARG2
                [[ 5 -eq 05 ]] && echo "5 equals 05"

ARG1 -ne ARG2   Returns true if ARG1 is not-equal to ARG2
                [[ 6 -ne 20 ]] && echo "6 is not equal to 20"

ARG1 -lt ARG2   Returns true if ARG1 is less-than to ARG2
                [[ 8 -lt 9 ]] && echo "8 is less than 9"
                [[ 3 < 4 ]] && echo "3 is less than 4"

ARG1 -le ARG2   Returns true if ARG1 is less-than-or-equal to ARG2
                [[ 3 -le 8 ]] && echo "3 is less than or equal to 8"

ARG1 -gt ARG2   Returns true if ARG1 is greater-than to ARG2
                [[ 5 -gt 10 ]] || echo "5 is not bigger than 10"
                [[ 4 > 2 ]] && echo "4 is greater than 2"

ARG1 -ge ARG2   Returns true if ARG1 is greater-than-or-equal to ARG2
                [[ 3 -ge 3 ]] && echo "3 is greater than or equal to 3"