# go scripting

拿go当脚本来用主要都是执行一些批处理的操作。虽然bash也可以完成这一类工作， perl或python也和go同样好用，但可能有一些人对golang比较熟悉，碰巧golang也能完成这一类工作，所以我们也可以在这方面尝试一下。
下面我们用golang来实现一个简单的业务布署功能。我有一个小项目叫做kentucky，它编译之后生成一个二进制文件。我需要将这个二进制文件在服务器上跑起来。做这个事情需要三个步骤：(1)结束服务器上正在运行的旧版本kentucky，(2)上传当前版本的kentucy到服务器，(3)启动新版本的kentucky。

## expect
expect是一个bash命令，用来实现自动和交互式任务进行通信，而无需人的干预。expect是不断发展的，随着时间的流逝，其功能越来越强大，已经成为系统管理员的的一个强大助手。

github上有一个项目使用golang提供了类似的功能，其地址是https://github.com/jamesharr/expect。

我们需要用expect来实现第一步和第三步。

下面的示例代码展示了如何实现第一步，杀死旧的进程：

    package main

    import (
    	"fmt"
    	"strings"
	    "time"

    	"github.com/jamesharr/expect"
    )

    func main() {
	    // Spawn an expect process
	    ssh, err := expect.Spawn("ssh", "root@zhangfuwen.com")
	    if err != nil {
		    fmt.Println(err)
		    return
	    }
	    ssh.SetTimeout(10 * time.Second)
	    const PROMPT = `.*#`

	    // Login
	    ssh.Expect(`[Pp]assword:`)
	    ssh.SendMasked("876400aa") // SendMasked hides from logging
	    ssh.Send("\n")
	    ssh.Expect(PROMPT) // Wait for prompt

	    // Run a command
	    ssh.SendLn("ps -ef | grep ' ./kentu' | awk '{ print $2 }'")
	    match, err := ssh.Expect(PROMPT) // Wait for prompt
	    results := strings.Split(match.Before, "\n")
	    var pid string
	    if len(results) >= 2 {
		    fmt.Println("pid is ", results[1])
		    pid = results[1]
	    } else {
		    fmt.Println("cannot find a process of this name")
		    return
	    }

	    ssh.SendLn("kill -9 " + pid)
	    match, err = ssh.Expect(PROMPT)
	    fmt.Println(match.Before)

	    // Wait for EOF
	    ssh.SendLn("exit")
	    ssh.ExpectEOF()
    }

以下代码实现第三步，即启动新的子服务进程：
```
func step3() { // start process
	// Spawn an expect process
	ssh, err := expect.Spawn("ssh", "root@zhangfuwen.com")
	if err != nil {
		fmt.Println(err)
		return
	}
	ssh.SetTimeout(10 * time.Second)
	const PROMPT = `.*#`

	// Login
	ssh.Expect(`[Pp]assword:`)
	ssh.SendMasked("876400aa") // SendMasked hides from logging
	ssh.Send("\n")
	ssh.Expect(PROMPT) // Wait for prompt

	// Run a command
	ssh.SendLn("cd /alidata/server/go1.4.2/go/gopath/src/kentucky")
	ssh.SendLn("nohup ./kentucky &")
	match, _ := ssh.Expect(PROMPT)
	fmt.Println(match.Before)

}```


