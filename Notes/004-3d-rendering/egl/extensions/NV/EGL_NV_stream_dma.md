# NV_stream_dma

Name

    NV_stream_dma

Name Strings

    EGL_NV_stream_dma

Contributors

    Santanu Thangaraj
    Daniel Kartch
    Arihant Jejani

Contacts

    Santanu Thangaraj, NVIDIA (sthangaraj 'at' nvidia.com)
    Arihant Jejani, NVIDIA (ajejani 'at' nvidia.com)

Status

    Draft

Version

    Version 1 - April 15, 2019

Number

    135

Extension Type

    EGL display extension

Dependencies

    Requires EGL_NV_stream_remote extension.
    
    Requires EGL_NV_stream_cross_system extension.
    
    Interacts with EGL_NV_stream_socket extensions.

Overview:

    This extension provides the framework for performing DMA transfers
    between opposite ends of a single stream, if there is no shared
    memory available between them. 
    
    In case of cross system streams the buffer contents of one end of
    the stream is transferred to other end using sockets by
    EGL_NV_stream_socket extension. Transferring buffer contents
    through sockets is slower compared to DMA transfers. Since DMA
    transfers have higher throughput compared to sockets, using 
    EGL_NV_stream_dma extension, applications can request EGL to
    utilize DMA channels to perform buffer copies.

New types

    None

New Procedures and functions

    None

New Tokens
    
    Accepted as attribute names in eglCreateStreamKHR,
    eglCreateStreamAttribKHR, eglQueryStreamKHR, and
    eglQueryStreamAttribKHR:

    EGL_STREAM_DMA_NV                         0x3371
    EGL_STREAM_DMA_SERVER_NV                  0x3372

Add to list of failures in section "3.10.1 Creating an EGLStream" in 
EGL_KHR stream:
    - EGL_BAD_MATCH is generated if the value of EGL_STREAM_DMA_NV is 
      neither EGL_TRUE nor EGL_FALSE.
    - EGL_BAD_MATCH is generated if the value of 
      EGL_STREAM_DMA_SERVER_NV is not EGL_DONT_CARE or a valid
      DMA server identifier as defined by the platform.

Add to "Table 3.10.4.4 EGLStream Attributes" in EGL_KHR_stream:

    Attribute                 Read/Write   Type              Section
    ------------------------ -----------   ------            ----------
    EGL_STREAM_DMA_NV            io       EGLint             3.10.4.x
    EGL_STREAM_DMA_SERVER_NV     io       platform dependent 3.10.4.x+1

Add new subsections to the end of section "3.10.4 EGLStream Attributes"
in EGL_KHR_stream:

    3.10.4.x EGL_STREAM_DMA_NV Attribute

    The EGL_STREAM_DMA_NV attribute may be set when the stream is 
    created, and indicates whether the DMA channels have to be used to
    transfer the buffer contents from producer to consumer. Legal 
    values are EGL_TRUE or EGL_FALSE. The default value is EGL_FALSE.

    A value of EGL_TRUE indicates that EGL has to use DMA channels to
    transfer buffers from producer to consumer.
    
    If EGL_FALSE is specified, DMA channels will not be utilized for
    buffer transfers.
    
    3.10.4.x+1 EGL_STREAM_DMA_SERVER_NV Attribute

    The EGL_STREAM_DMA_SERVER_NV attribute is a platform dependent
    identifier which may be set when the stream is created and it
    indicates the server, which must be contacted to handle DMA 
    transfers, if that server is not local. Legal values, aside from
    EGL_DONT_CARE, are determined by the implementation. The default
    value is EGL_DONT_CARE.
    
Issues

    1.  What happens when application requests DMA copy using 
        EGL_STREAM_DMA_NV attribute in eglCreateStreamKHR or 
        eglCreateStreamAttribKHR API, but the system does not support 
        access to DMA channels?

        RESOLVED: The functions return EGL_NO_STREAM_KHR and 
        EGL_BAD_ATTRIBUTE error is set.
        
    2.  What happens when application requests DMA copy using 
        EGL_STREAM_DMA_NV attribute in eglCreateStreamKHR or 
        eglCreateStreamAttribKHR API, and the system supports DMA 
        channels, but currently none of the channels are available for
        use?
        
        RESOLVED: The functions return EGL_NO_STREAM_KHR and
        EGL_BAD_ATTRIBUTE error is set.
        
Revision History

    #1  (April 15, 2019) Santanu Thangaraj
        - Initial version
