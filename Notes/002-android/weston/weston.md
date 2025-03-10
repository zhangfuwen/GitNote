```puml
participant client
participant weston

client -> weston : surface = wl_compositor_create_surface()
client -> client++ : fd = memfd_create ()
client -> weston : pool = new pool(fd) 
client -> client : buffer = wl_shm_pool_create_buffer(pool) 
client -> weston : wl_surface_attach(surface, buffer)
client -> weston : wl_surface_damage(region)
client -> weston : wl_surface_commit(surface)

```

应用商店关键路径
```puml
actor User
participant AppStore
participant LotClient
participant LaunchPad
participant Dock

autonumber 00 "<b>[000]"
User -> AppStore++:  点击安装按钮
AppStore -> LotClient++ : download()
LotClient -> LotClient++: Notification, 下载，安装, 关闭Notification
LotClient -> LaunchPad++ : create shortcut with icon\n目前<color:green>可创建</color>，但\n<color:red>图标显示不对</color>
LaunchPad--
LotClient--
LotClient--

autonumber 10 "<b>[000]"
User -> LaunchPad++ : 点击shortcut图标
LaunchPad -> Dock++: NewTask启动对应shortcut图标的Activity
Dock -> Dock : <color:green>显示为独立icon</color>\n <color:red>图标不对</color>
LaunchPad--
Dock--

autonumber 20 "<b>[000]"
User -> LaunchPad++: 长按图标，选卸载(<color:green>已完成</color>)
LaunchPad -> LaunchPad : <color:green>不删除图标，显示Toast“卸载中”</color>
LaunchPad -> LotClient++ :发送broadcast(<color:green>ok</color>)
LotClient -> LotClient++ : 执行卸载
alt 卸载成功
LotClient -> LaunchPad : <color:green>发送broadcast</color>
LaunchPad -> LaunchPad : <color:green>删除图标</color>
else
LotClient -> LaunchPad : <color:green>发送失败broadcast</color>
LaunchPad -> LaunchPad : <color:green>显示安装失败, </color> \n<color:green>保留图标，并支持再次删除</color>
end






```