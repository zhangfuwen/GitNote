# spring devtools

用于开发时即时编译和刷新。

## 安装
Maven. 
```
<dependencies>
	<dependency>
		<groupId>org.springframework.boot</groupId>
		<artifactId>spring-boot-devtools</artifactId>
		<optional>true</optional>
	</dependency>
</dependencies>
```

Gradle. 
```
dependencies {
	compile("org.springframework.boot:spring-boot-devtools")
}
```

## 注意事项

1. 运行打包好的可执行程序，如java -jar xxx.jar时，devtools不运行，它认为这时是生产环境
2. 为防止devtools被传递给使用的模块的其他工程，可以在maven中设置optional，或是gradle中设置compileOnly
3. 重新打包发布的时候devtools不会包含在里面，要是想使用它的远程调试功能，需要设置excludeDevtools为关闭状态。这样就可以包含了。

## 缓存功能的开启和关闭

正常是在application.properties中设置，但使用了devtools就需要设置了，devtools会自动判断是研发还是生产环境，并自动设置这些项。
比如：spring.thymeleaf.cache

支持的有：
1. FreeMarker
2. Groovy
3. Thymeleaf
4. Mustache

完整支持列表见：
https://github.com/spring-projects/spring-boot/blob/v2.0.2.RELEASE/spring-boot-project/spring-boot-devtools/src/main/java/org/springframework/boot/devtools/env/DevToolsPropertyDefaultsPostProcessor.java

## 自动重启

classpath内的文件变动就会触发重启，devtools通过重启使变动生效。但资源文件变动不触动重启，他们只需要不缓存就行了。

Triggering a restart

As DevTools monitors classpath resources, the only way to trigger a restart is to update the classpath. The way in which you cause the classpath to be updated depends on the IDE that you are using. In Eclipse, saving a modified file causes the classpath to be updated and triggers a restart. In IntelliJ IDEA, building the project (**Build -> Build Project**) has the same effect.

## 版本

http://repo.maven.apache.org/maven2/org/springframework/boot/spring-boot-devtools/



没读完：
http://www.cnblogs.com/java-zhao/p/5502398.html
http://ju.outofmemory.cn/entry/241222

https://docs.spring.io/spring-boot/docs/current/reference/html/using-boot-devtools.html
