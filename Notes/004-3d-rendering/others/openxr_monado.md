```plantuml


class oxr_session {
	(*create_swapchain)
}

class oxr_session {
	.create_swapchain=oxr_swapchain_gl_create
}


```
``` 
oxr_session_populate_egl -> xrt_gfx_provider_create_gl_egl

eglGenTextures
eglGetNativeClientBufferANDROID
eglCreateImageKHR

client_gl_eglimage_swapchain_create

	(*create_swapchain) = oxr_swapchain_gl_create ->
}
```



```plantuml


```