# github pages的用法

以在这个仓的流程直接推送markdown给master分支，然后gh-pages bot直接从master分支捞md文件，用jekyll构建成静态页面用于显示。

这个流程有一个问题，gh-pages bot从master分支捞md文件并用jekyll构建主个过程不是通过yml文件控制的，而是gh-pages bot自己的行为。

现在改成新的流程，先推到代码到master分支，然后自己创建一个github actions任务做一些基本的处理，然后上传到gh-pages分支，让gh-pages bot从gh-pages 分支取文件并处理。