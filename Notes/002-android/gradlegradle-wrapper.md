# Gradle及Gradle Wrapper的使用

## 1. Gradle及gradle wrapper介绍

gradle和maven一样，是java开发中经常用到的构建管理工具。它的主要功能包括（1）包管理和（2）生命周期管理。
`包管理`：在网站比如说bintray上java的几乎所有开源组件的几乎所有版本的jar包及其描述信息。想在项目中使用某个jar包的时候，只需要把包名和版本号写在build.gradle的dependency里面即可使用。执行grade build时，自动从网站上下载jar供编译。
gradle支持maven repository，所以不用担心gradle的组件种类没有maven丰富。
`生命周期管理`:支持依赖库下载,编译，打包等整个软件发布过程的自动化。

### 1.1 gradle wrapper
Gradle是个好东西，但是你发布一个工程，别人要构建你的工程还需要先下载正确版本的gradle，这就是一个很麻烦的事情。直接把gradle一起发布则源码包又太大。因而有了gradle wrapper，它包括一个jar包，一个.properties配置文件和两个分别在*nix和windows下使用的脚本gradlew和gradlew.bat。
构建时执行脚本gradlew或gradle.bat build，脚本就是自动检测java环境，下载正确版本的gradle，然后把构建参数（比如说这里的build)原样传给gradle去执行。

## 2. gradle wrapper详细介绍

gradle wrapper文件结构：
```

Project-name/
  #gradel wrapper文件
  gradlew
  gradlew.bat
  gradle/wrapper/
    gradle-wrapper.jar
    gradle-wrapper.properties
  
  #gradle项目文件
  app
  build.gradle
  settings.gradle
```
### 2.1 gradle wrapper下载gradle
四个文件中三个是不可以修改的，可以修改的是gradle-wrapper.properties这个文本文件。
它的内容大致如下：


```
#Thu Mar 02 23:08:16 CST 2017
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
distributionUrl=https\://services.gradle.org/distributions/gradle-3.3-all.zip
```
这里描述的内容是下载gradle的方式，即从https\://services.gradle.org/distributions/gradle-3.3-all.zip
下载一个zip文件，解压到wrapper/dists目录下。
对于中国以及公司内网的网友来说，下载并不容易。如果你下载出问题了，可以用如下方式解决：

#### 2.1.1 gradle下载方式1：相对路径

最简单的方式是手工下载好，然后以相对路径的方式加在这里。正常情况下，你应该用这种方法。相对路径是相对于properties文件的。比如说放在同级文件夹下：

```
distributionUrl=./gradle-3.3-all.zip
```
#### 2.1.2 gradle下载方式2：使用proxy
在gradle.properties里面写入：

```
systemProp.http.proxyHost=www.somehost.org
systemProp.http.proxyPort=8080
systemProp.http.proxyUser=userid
systemProp.http.proxyPassword=password
systemProp.http.nonProxyHosts=*.nonproxyrepos.com|localhost
```

或


```
systemProp.https.proxyHost=www.somehost.org
systemProp.https.proxyPort=8080
systemProp.https.proxyUser=userid
systemProp.https.proxyPassword=password
systemProp.https.nonProxyHosts=*.nonproxyrepos.com|localhost
```

或者你使用域账号，则无需手动填加用户名和密码则这两个字段可以省。

### 2.2 生成gradle wrapper
生成gradle wrapper只要一个命令：


```
gradle wrapper --gradle-version 2.0
```
这里的版本号会体现在gradle-wrapper.properties这个文件里。

