# EXT_external_buffer

Name

    EXT_external_buffer

Name Strings

    GL_EXT_external_buffer

Contact

    Jeff Leger (jleger 'at' qti.qualcomm.com)

Contributors

    Sam Holmes
    Maurice Ribble
    Matt Netsch
    Jeremy Gebben
    John Bates
    Craig Donner
    Jeff Leger
    Rob VanReenen
    Tom Kneeland
    Jesse Hall
    Jan-Harald Fredriksen
    Daniel Koch
    Mathias Heyer

Status

    Complete

Version

    Last Modified Date: May 29, 2017
    Revision: 1.0

Number

    OpenGL ES Extension #284
    OpenGL Extension #508

Dependencies

    OpenGL ES 3.1 and EXT_buffer_storage are required for OpenGL ES
    implementations.

    OpenGL 4.4 is required for OpenGL implementations.

    This extension is written against the OpenGL ES 3.1 (June 4, 2014)
    Specification.

    This extension is written against version 3 of EXT_buffer_storage.

    The definition of this extension is affected by the presence of
    GL_EXT_direct_state_access, GL_ARB_direct_state_access, or OpenGL 4.5.


Overview

    Extension EXT_buffer_storage introduced immutable storage buffers to
    OpenGL ES. This extension allows the data store for an immutable buffer to
    be sourced from an external EGLClientBuffer, allowing sharing of EGL client
    buffers across APIs, across processes, and across different processing
    cores such as the GPU, CPU, and DSP.

    Operations can then be performed on the external buffer using standard
    GL buffer object procedures. The data in the allocation is not copied to
    the buffer object's data store; the external allocation represents a single
    memory allocation that can be shared across multiple GL objects -- this
    aspect is similar to EGL external images.  On the other hand, the external
    buffer does not provide lifetime guarantees including orphaning and sibling
    behavior as provided by EGL external images.

    The EGLClientBuffer must be allocated in a way which permits this shared
    access. For example, on Android via a shareable Android hardware buffer.
    This extension does not enable support for arbitrary EGLClientBuffers to be
    used as an external buffer.

    It is the application's responsibility to ensure synchronization between
    operations performed by separate components (DSP / CPU / GPU) and processes
    on the external buffer. Additionally the application is responsible for
    avoiding violating existing GL spec requirements. For example, mapping a
    single shared allocation to two GL buffer objects and then performing
    CopyBufferSubData such that the read and write regions overlap would
    violate the existing CopyBufferSubData spec regarding copies performed
    with the same buffer set for source and destination.

    The application must take any steps necessary to ensure memory access to
    the external buffer behaves as required by the application. For example,
    preventing compilation differences in data padding from causing data to be
    inadvertently corrupted by using defined structure alignment methods such
    as the std140 layout qualifier.  The application is responsible for
    managing the lifetime of the external buffer, ensuring that the external
    buffer is not deleted as long as there are any GL buffer objects referring
    to it.

New Types

    /*
     * GLeglClientBufferEXT is an opaque handle to an EGLClientBuffer
     */
    typedef void* GLeglClientBufferEXT;

New Procedures and Functions

    void BufferStorageExternalEXT(enum               target,
                                  intptr             offset,
                                  sizeiptr           size,
                                  eglClientBufferEXT clientBuffer,
                                  bitfield           flags);

    [[ The following is only added if GL_EXT_direct_state_access,
       GL_ARB_direct_state_access, or OpenGL 4.5 is supported. ]]

    void NamedBufferStorageExternalEXT(uint               buffer,
                                       intptr             offset,
                                       sizeiptr           size,
                                       eglClientBufferEXT clientBuffer,
                                       bitfield           flags);

New Tokens

    None

Additions to Chapter 6 of the OpenGL ES 3.1 Specification (Buffer Objects)

Modify Section 6.2, (Creating and Modifying Buffer Object Data Stores).  After
the section describing BufferStorageEXT, insert the following:

    The command

        void BufferStorageExternalEXT(enum target, intptr offset,
             sizeiptr size, eglClientBufferEXT clientBuffer,
             bitfield flags);

    behaves similar to BufferStorageEXT, but rather than allocate an immutable
    data store, the specified client buffer is referenced as the immutable
    data store.  Such a store may not be modified through further calls to
    BufferStorageExternalEXT, BufferStorageEXT, or BufferData.

    <target> Specifies the target buffer object. The symbolic constant must be
    one of the targets listed in table 6.1.  <offset> and <size> specify, in
    basic machine units, the range of the client buffer to be bound to the data
    store. <offset> must be zero.

    <clientBuffer> Is the handle of a valid EGLClientBuffer resource (cast
    into type eglClientBufferEXT).  The EGLClientBuffer must be allocated in a
    platform-specific way which permits shared access.  For example, on Android
    via a sharable Android hardware buffer (struct AHardwareBuffer), converted
    into EGLClientBuffer via extension EGL_ANDROID_get_native_client_buffer.
    Other platforms would require a similar mechanism. This extension does not
    enable support for arbitrary EGLClientBuffers to be used as a shared buffer.
    <flags> is the bitwise OR of flags describing the intended usage of the buffer
    object's external data store by the application. Valid flags and their
    meanings are as described for BufferStorageEXT.

    The values of the buffer object's state variables will match those for other
    *BufferStorageEXT calls, as specified in table 6.3.

    The behavior follows other immutable buffers; BufferStorageExternalEXT sets the
    created buffer's BUFFER_IMMUTABLE_STORAGE_EXT to TRUE.

    [[ The following is only added if GL_EXT_direct_state_access,
       GL_ARB_direct_state_access, or OpenGL 4.5 is supported. ]]

    The command

        void NamedBufferStorageExternalEXT(uint buffer, intptr offset,
             sizeiptr size, eglClientBufferEXT clientBuffer,
             bitfield flags);

    behaves similarly to BufferStorageExternalEXT, except that the buffer whose
    storage is to be defined is specified by <buffer> rather than by the current
    binding to <target>.


Errors

    INVALID_OPERATION is generated by BufferStorageExternalEXT if zero is bound to
    <target>.

    INVALID_OPERATION is generated by BufferStorageExternalEXT, if the
    BUFFER_IMMUTABLE_STORAGE flag of the buffer bound to <target> is TRUE.

    INVALID_VALUE is generated by BufferStorageExternalEXT if <offset> is not 0.

    INVALID_VALUE is generated by BufferStorageExternalEXT if <size> is 0
    or negative.

    INVALID_VALUE is generated by BufferStorageExternalEXT if <offset> + <size>
    exceeds the size of the EGLClientBuffer.

    INVALID_VALUE is generated by BufferStorageExternalEXT if <flags> has any
    bits set other than those defined above.

    INVALID_VALUE is generated by BufferStorageExternalEXT if <flags> contains
    MAP_PERSISTENT_BIT_EXT but does not contain at least one of MAP_READ_BIT or
    MAP_WRITE_BIT.

    INVALID_VALUE is generated by BufferStorageExternalEXT if <flags> contains
    MAP_COHERENT_BIT_EXT, but does not also contain MAP_PERSISTENT_BIT_EXT.

    INVALID_ENUM is generated by BufferStorageExternalEXT if <target> is not one
    of the accepted buffer targets.

    INVALID_OPERATION is generated by BufferStorageExternalEXT if the shared
    buffer is not allocated in a way which permits shared access by the GPU.

    [[ The following is only added if GL_EXT_direct_state_access or
       GL_ARB_direct_state_access is supported. ]]

    An INVALID_OPERATION error is generated by NamedBufferStorageExternalEXT if
    the BUFFER_IMMUTABLE_STORAGE_EXT flag of <buffer> is set to TRUE.

Interactions with GL_EXT_direct_state_access, GL_ARB_direct_state_access and
OpenGL 4.5

    If none of GL_EXT_direct_state_access, GL_ARB_direct_state_access, or
    OpenGL 4.5, the NamedBufferStorageExternalEXT entry-point is not
    added and all references to it should be ignored.

Issues

    1. How are possible GPU cache interactions handled?

    The application is responsible for synchronizing writes to the shared buffer
    by other processing cores (e.g. DSP), and making those available to CPU
    reads for the processing of client-side GL commands (e.g., BufferSubData).
    The GL implementation should guarantee that available writes by other cores
    (e.g., DSP) are visible to the GPU when server-side commands read from the
    shared buffer.

    PROPOSED: The exact granularity with which available writes from other cores
    e.g., DSP) become visible to the CPU and GPU is implementation dependent.

    2. Should EGLClientBuffers, be directly referenced by the GL API?

    For images, a set of EGL and client API extensions provide import/export
    of EGLImages from client APIs and native buffers.  The EGLImage also provides
    lifetime guarantees including orphaning and sibling behavior.  This extension
    is more narrowly focused, specifically targeted to the import of EGLClientBuffers
    as GL buffers, and requiring the application to manage the resource lifetime.
    As such, it may not warrant a new EGL object or EGL extension.

    RESOLVED:  A corresponding EGL object and extension is not required.  When
    using this extension, applications are expected to cast EGLClientBuffer as
    GLeglClientBufferEXT.

Revision History

      Rev.    Date      Author    Changes
      ----  ----------  --------  -----------------------------------------
      0.1   04/18/2017  sholmes   Initial version. Based on QCOM_shared_buffer.
      0.2   05/16/2017  jleger    Renamed the extension and reworked it to to
                                  be an extension to EXT_buffer_storage.
      0.3   05/24/2017  jleger    Add offset parameter and other cleanup.
      0.4   05/25/2017  jleger    Add DSA entrypoint and minor cleanup.
      1.0   05/29/2017  dgkoch    Add interactions with GL, minor cleanup.
