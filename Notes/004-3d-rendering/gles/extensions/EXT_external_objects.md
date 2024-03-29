# EXT_external_objects

Name

    EXT_external_objects

Name Strings

    GL_EXT_memory_object
    GL_EXT_semaphore

Contributors

    Carsten Rohde, NVIDIA
    Dave Airlie, Red Hat
    James Jones, NVIDIA
    Jan-Harald Fredriksen, ARM
    Jeff Juliano, NVIDIA
    Michael Worcester, Imagination Technologies

Contact

    James Jones, NVIDIA (jajones 'at' nvidia.com)

Status

    Complete

Version

    Last Modified Date: July 18, 2018
    Revision: 14

Number

    503
    OpenGL ES Extension #280

Dependencies

    Written against the OpenGL 4.5 and OpenGL ES 3.2 specifications.

    GL_EXT_memory_object requires ARB_texture_storage or a version of
    OpenGL or OpenGL ES that incorporates it.

    GL_EXT_semaphore requires OpenGL 1.0.

    ARB_direct_state_access (OpenGL) interacts with GL_EXT_memory_object
    when OpenGL < 4.5 is used.

    ARB_sparse_texture (OpenGL) interacts with GL_EXT_memory_object

    EXT_sparse_texture (OpenGL ES) interacts with GL_EXT_memory_object

    EXT_protected_textures (OpenGL ES) interacts with GL_EXT_memory_object

Overview

    The Vulkan API introduces the concept of explicit memory objects and
    reusable synchronization objects.  This extension brings those
    concepts to the OpenGL API via two new object types:

       Memory objects
       Semaphores

    Rather than allocating memory as a response to object allocation,
    memory allocation and binding are two separate operations in Vulkan.
    This extension allows an OpenGL application to import a Vulkan
    memory object, and to bind textures and/or buffer objects to it.

    No methods to import memory objects are defined here.  Separate
    platform-specific extensions are defined for this purpose.

    Semaphores are synchronization primitives that can be waited on and
    signaled only by the GPU, or in GL terms, in the GL server.  They
    are similar in concept to GL's "sync" objects and EGL's "EGLSync"
    objects, but different enough that compatibilities between the two
    are difficult to derive.

    Rather than attempt to map Vulkan semaphores on to GL/EGL sync
    objects to achieve interoperability, this extension introduces a new
    object, GL semaphores, that map directly to the semantics of Vulkan
    semaphores.  To achieve full image and buffer memory coherence with
    a Vulkan driver, the commands that manipulate semaphores also allow
    external usage information to be imported and exported.

New Procedures and Functions

    The following commands are added if either of the GL_EXT_memory_object
    or GL_EXT_semaphore strings are reported:

        void GetUnsignedBytevEXT(enum pname,
                                 ubyte *data);

        void GetUnsignedBytei_vEXT(enum target,
                                   uint index,
                                   ubyte *data);

    If the GL_EXT_memory_object string is reported, the following
    commands are added:

        void DeleteMemoryObjectsEXT(sizei n,
                                    const uint *memoryObjects);

        boolean IsMemoryObjectEXT(uint memoryObject);

        void CreateMemoryObjectsEXT(sizei n,
                                    uint *memoryObjects);

        void MemoryObjectParameterivEXT(uint memoryObject,
                                        enum pname,
                                        const int *params);

        void GetMemoryObjectParameterivEXT(uint memoryObject
                                           enum pname,
                                           int *params);

        void TexStorageMem2DEXT(enum target,
                                sizei levels,
                                enum internalFormat,
                                sizei width,
                                sizei height,
                                uint memory,
                                uint64 offset);

        void TexStorageMem2DMultisampleEXT(enum target,
                                           sizei samples,
                                           enum internalFormat,
                                           sizei width,
                                           sizei height,
                                           boolean fixedSampleLocations,
                                           uint memory,
                                           uint64 offset);

        void TexStorageMem3DEXT(enum target,
                                sizei levels,
                                enum internalFormat,
                                sizei width,
                                sizei height,
                                sizei depth,
                                uint memory,
                                uint64 offset);

        void TexStorageMem3DMultisampleEXT(enum target,
                                           sizei samples,
                                           enum internalFormat,
                                           sizei width,
                                           sizei height,
                                           sizei depth,
                                           boolean fixedSampleLocations,
                                           uint memory,
                                           uint64 offset);

        void BufferStorageMemEXT(enum target,
                                 sizeiptr size,
                                 uint memory,
                                 uint64 offset);

        [[ The following are added if direct state access is supported ]]

        void TextureStorageMem2DEXT(uint texture,
                                    sizei levels,
                                    enum internalFormat,
                                    sizei width,
                                    sizei height,
                                    uint memory,
                                    uint64 offset);

        void TextureStorageMem2DMultisampleEXT(uint texture,
                                               sizei samples,
                                               enum internalFormat,
                                               sizei width,
                                               sizei height,
                                               boolean fixedSampleLocations,
                                               uint memory,
                                               uint64 offset);

        void TextureStorageMem3DEXT(uint texture,
                                    sizei levels,
                                    enum internalFormat,
                                    sizei width,
                                    sizei height,
                                    sizei depth,
                                    uint memory,
                                    uint64 offset);

        void TextureStorageMem3DMultisampleEXT(uint texture,
                                               sizei samples,
                                               enum internalFormat,
                                               sizei width,
                                               sizei height,
                                               sizei depth,
                                               boolean fixedSampleLocations,
                                               uint memory,
                                               uint64 offset);

        void NamedBufferStorageMemEXT(uint buffer,
                                      sizeiptr size,
                                      uint memory,
                                      uint64 offset);

        [[ The following are available in OpenGL only ]]

        void TexStorageMem1DEXT(enum target,
                                sizei levels,
                                enum internalFormat,
                                sizei width,
                                uint memory,
                                uint64 offset);

        [[ The following are availble in OpenGL only, and only when
           direct state access is available ]]

        void TextureStorageMem1DEXT(uint texture,
                                    sizei levels,
                                    enum internalFormat,
                                    sizei width,
                                    uint memory,
                                    uint64 offset);

    If the GL_EXT_semaphore string is reported, the following
    commands are added:

        void GenSemaphoresEXT(sizei n,
                              uint *semaphores);

        void DeleteSemaphoresEXT(sizei n,
                                 const uint *semaphores);

        boolean IsSemaphoreEXT(uint semaphore);

        void SemaphoreParameterui64vEXT(uint semaphore,
                                        enum pname,
                                        const uint64 *params);

        void GetSemaphoreParameterui64vEXT(uint semaphore,
                                           enum pname,
                                           uint64 *params);

        void WaitSemaphoreEXT(uint semaphore,
                              uint numBufferBarriers,
                              const uint *buffers,
                              uint numTextureBarriers,
                              const uint *textures,
                              const GLenum *srcLayouts);

        void SignalSemaphoreEXT(uint semaphore,
                                uint numBufferBarriers,
                                const uint *buffers,
                                uint numTextureBarriers,
                                const uint *textures,
                                const GLenum *dstLayouts);

New Tokens

    If the GL_EXT_memory_object string is reported, the following tokens are
    added:

    Accepted by the <pname> parameter of TexParameter{ifx}{v},
    TexParameterI{i ui}v, TextureParameter{if}{v}, TextureParameterI{i ui}v,
    GetTexParameter{if}v, GetTexParameterI{i ui}v, GetTextureParameter{if}v,
    and GetTextureParameterI{i ui}v:

        TEXTURE_TILING_EXT                         0x9580

    Accepted by the <pname> parameter of MemoryObjectParameterivEXT, and
    GetMemoryObjectParameterivEXT:

        DEDICATED_MEMORY_OBJECT_EXT                0x9581

        [[ The following are available when GL_EXT_protected_textures is
           available ]]

	PROTECTED_MEMORY_OBJECT_EXT                0x959B

    Accepted by the <pname> parameter of GetInternalFormativ or
    GetInternalFormati64v:

        NUM_TILING_TYPES_EXT                       0x9582
        TILING_TYPES_EXT                           0x9583

    Returned in the <params> parameter of GetInternalFormativ or
    GetInternalFormati64v when the <pname> parameter is TILING_TYPES_EXT,
    returned in the <params> parameter of GetTexParameter{if}v,
    GetTexParameterI{i ui}v, GetTextureParameter{if}v, and
    GetTextureParameterI{i ui}v when the <pname> parameter is
    TEXTURE_TILING_EXT, and accepted by the <params> parameter of
    TexParameter{ifx}{v}, TexParameterI{i ui}v, TextureParameter{if}{v},
    TextureParameterI{i ui}v when the <pname> parameter is
    TEXTURE_TILING_EXT:

        OPTIMAL_TILING_EXT                         0x9584
        LINEAR_TILING_EXT                          0x9585

    The following tokens are added if either of the GL_EXT_memory_object or
    GL_EXT_semaphore strings are reported:

    Accepted by the <pname> parameter of GetBooleanv, GetDoublev, GetFloatv,
    GetIntegerv, GetInteger64v, GetUnsignedBytevEXT, and the <target>
    parameter of GetBooleani_v, GetIntegeri_v,GetFloati_v, GetDoublei_v,
    GetInteger64i_v, and GetUnsignedBytei_vEXT:

        NUM_DEVICE_UUIDS_EXT                       0x9596
        DEVICE_UUID_EXT                            0x9597
        DRIVER_UUID_EXT                            0x9598

    Constant values:

        UUID_SIZE_EXT                              16

    If the GL_EXT_semaphore string is reported, the following tokens are
    added:

    Accepted by the <dstLayouts> parameter of SignalSemaphoreEXT and the
    <srcLayouts> parameter of WaitSemaphoreEXT:

        LAYOUT_GENERAL_EXT                            0x958D
        LAYOUT_COLOR_ATTACHMENT_EXT                   0x958E
        LAYOUT_DEPTH_STENCIL_ATTACHMENT_EXT           0x958F
        LAYOUT_DEPTH_STENCIL_READ_ONLY_EXT            0x9590
        LAYOUT_SHADER_READ_ONLY_EXT                   0x9591
        LAYOUT_TRANSFER_SRC_EXT                       0x9592
        LAYOUT_TRANSFER_DST_EXT                       0x9593
        LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_EXT 0x9530
        LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_EXT 0x9531

Additions to Chapter 2 of the OpenGL 4.5 Specification (OpenGL
Fundamentals)

    Add two new sections after 2.6.13 (Sync Objects)

        2.6.14 Semaphore Objects

        A /semaphore object/ is a synchronization primitive similar to a
        /sync object/, but with semantics based on Vulkan semaphores.

        Semaphore objects may be shared.  They are described in detail in
        section 4.2.

        2.6.15 Memory Objects

        Many GL objects have some associated data stored in GL server
        memory.  /Memory objects/ are an abstract representation of GL
        server memory suitable for use as the backing store of a
        /buffer object/, a /texture object/, or both, depending on how
        the memory referred to by the object was allocated.  Memory
        objects can not be created directly within the GL.  They must be
        imported from an API capable of allocating abstract memory, such
        as Vulkan.

        Memory objects may be shared.  They are described in detail in
        Chapter 6 (Memory Objects).

Additions to Chapter 4 of the OpenGL 4.5 Specification (Event Model)

    Add a new section between sections 4.1, "Sync Objects and Fences"
    and section 4.2, "Query Objects and Asynchronous Queries"

        4.2 Semaphore Objects

        Like sync objects, a semaphore object acts as a /synchronization
        primitive/.  However, semaphore objects differ from sync objects
        in several ways:

        * They may only be created by importing an external semaphore
          handle into the GL.

        * They are reusable.

        * As a corollary to the above behavior, separate commands are
          provided to create and signal semaphore objects.

        * Their state is reset upon completion of a wait operation.

        * As a corollary to the above behavior, only a single waiter may
          be associated with a unique signal command.

        * There is no way to wait for a semaphore to become signaled in
          the GL client.  All waits operations execute in the GL server,
          and semaphores have no queryable state.

        The command

            void GenSemaphoresEXT(sizei n,
                                  uint *semaphores);

        returns <n> previous unused semaphore names in <semaphores>.
        These names are marked as used, for the purposes of
        GenSemaphoresEXT only, but they are associated with semaphore
        state only when an external semaphore handle is imported to
        them.

        Semaphore objects are deleted by calling

            void DeleteSemaphoresEXT(sizei n,
                                     const uint *semaphores);

        <semaphores> contains <n> names of semaphores to be deleted.
        After a semaphore is deleted, it unreferences any external
        semaphore state it referenced, and its name is again unused.
        Unused names in <semaphores> are silently ignored, as is the
        value zero.

        The command

            boolean IsSemaphoreEXT(uint semaphore);

        returns TRUE if <semaphore> is the name of a semaphore.  If
        <semaphore> is zero, or if <semaphore> is a non-zero value that
        is not the name of a semaphore, IsSemaphore returns FALSE.

        4.2.1 Importing External Semaphore Handles into Semaphores

        A semaphore is created by importing an external semaphore object
        via a reference to its associated external handle.  The
        supported set of external handle types and associated import
        functions are listed in table 4.2.

        Table 4.2: Commands for importing external semaphore handles.

        | Handle Type | Import command |
        +-------------+----------------+
        +-------------+----------------+

        Applications must only import external semaphore handles exported
        from the same device or set of devices used by the current context,
        and from compatible driver versions.  To determine which devices are
        used by the current context, first call GetIntegerv with <pname> set
        to NUM_DEVICE_UUIDS_EXT, then call GetUnsignedBytei_vEXT with <target>
        set to DEVICE_UUID_EXT, <index> set to a value in the range [0,
        <number of device UUIDs>), and <data> set to point to an array of
        UUID_SIZE_EXT unsigned bytes.  To determine the driver ID of the
        current context, call GetUnsignedBytevEXT with <pname> set to
        DRIVER_UUID_EXT and <data> set to point to an array of UUID_SIZE_EXT
        unsigned bytes.

        These device and driver ID values can be used to correlate devices
        and determine driver compatibility across process and API boundaries.

        External handles are often defined using platform-specific
        types.  Therefore, the base GL specification defines no methods
        to import an external handle.

        4.2.2 Semaphore Parameters

        Semaphore parameters control how semaphore wait and signal
        operations behave.  Table 4.3 defines which parameters are available
        for a semaphore based on the external handle type from which it was
        imported.  Semaphore parameters are set using the command

            void SemaphoreParameterui64vEXT(uint semaphore,
                                            enum pname,
                                            const uint64 *params);

        <semaphore> is the name of the semaphore object on which the
        parameter <pname> will be set to the value(s) in <pname>.

        Table 4.3: Semaphore parameters

        | Name | External Handle Types | Legal Values |
        +------+-----------------------+--------------+
        +------+-----------------------+--------------+

        Parameters of a semaphore object may be queried with the command

            void GetSemaphoreParameterui64EXT(uint semaphore,
                                              enum pname,
                                              uint64 *params);

        <semaphore> is the semaphore object from with the parameter <pname>
        is queried.  The value(s) of the parameter are returned in <params>.
        <pname> may be any value in table 4.3.

        4.2.3 Waiting for Semaphores

        The command

            void WaitSemaphoreEXT(uint semaphore,
                                  uint numBufferBarriers,
                                  const uint *buffers,
                                  uint numTextureBarriers,
                                  const uint *textures,
                                  const GLenum *srcLayouts);

        Returns immediately but causes GL server to block until
        <semaphore> is signaled.  If an error occurs, WaitSemaphore
        generates a GL error as specified below, and does not cause the
        GL server to block.

        After completion of the semaphore wait operation, the semaphore
        will be returned to the unsignaled state.  Calling WaitSemaphore on
        a semaphore that has not previously had a signal operation flushed
        to the GL server or submitted by an external semaphore signaler
        since the semaphore was created or last waited on results in
        undefined behavior.

        Following completion of the semaphore wait operation, memory will
        also be made visible in the specified buffer and texture objects.
        Since texture layout state is managed internally by the GL, but may
        have been modified by an external API, the current layout of the
        textures must be specified to initialize internal GL state prior to
        using the textures after an external access.  The valid layouts
        correspond to those specified by the Vulkan API, as described in
        table 4.4.  However, the layouts do not necessarily correspond to an
        optimal state for any particular GL operation.  The GL will simply
        perform appropriate transitions internally as necessary based on the
        specified current layout of the texture.

        Table 4.4: Texture layouts and corresponding Vulkan Image Layouts

        | Texture Layout                                   | Equivalent Vulkan Image Layout                                 |
        +--------------------------------------------------+----------------------------------------------------------------+
        | GL_NONE                                          | VK_IMAGE_LAYOUT_UNDEFINED                                      |
        | GL_LAYOUT_GENERAL_EXT                            | VK_IMAGE_LAYOUT_GENERAL                                        |
        | GL_LAYOUT_COLOR_ATTACHMENT_EXT                   | VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL                       |
        | GL_LAYOUT_DEPTH_STENCIL_ATTACHMENT_EXT           | VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT                       |
        | GL_LAYOUT_DEPTH_STENCIL_READ_ONLY_EXT            | VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL                |
        | GL_LAYOUT_SHADER_READ_ONLY_EXT                   | VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL                       |
        | GL_LAYOUT_TRANSFER_SRC_EXT                       | VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL                           |
        | GL_LAYOUT_TRANSFER_DST_EXT                       | VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL                           |
        | GL_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_EXT | VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL_KHR |
        | GL_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_EXT | VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL_KHR |
        +-------------------------------------------------------------------------------------------------------------------+

        4.2.4 Signaling Semaphores

        The command

            void SignalSemaphoreEXT(uint semaphore,
                                    uint numBufferBarriers,
                                    const uint *buffers,
                                    uint numTextureBarriers,
                                    const uint *textures,
                                    const GLenum *dstLayouts);

        will insert a semaphore signaling operation in the GL command
        stream.

        Prior to signaling the semaphore, memory used by the specified
        buffer objects and textures will be made visible, and textures
        can be transitioned to a specified internal layout to allow
        applications to access the textures using a consistent layout in
        an external API or process.  Possible layouts are specified in
        table 4.3, along with their corresponding layout in the Vulkan
        API.

Add a new Chapter, "Memory Objects", between Chapter 5 (Shared Objects
and Multiple Contexts) and Chapter 6 (Buffer Objects)

    Memory objects reference a fixed-size allocation of abstract server
    memory.  The memory may not be accessed directly, but may be bound
    to other objects that require a data store in server memory.  The
    memory itself is allocated outside the scope of the GL, and is
    merely referenced by a memory object.

    The command

       void CreateMemoryObjectsEXT(sizei n, uint *memoryObjects);

    returns <n> previously unused memory object names in <memoryObjects>.
    The memory objects named contain default state, but initially have no
    external memory associated with them.

    Memory objects are deleted by calling

        void DeleteMemoryObjectsEXT(sizei n, const uint *memoryObjects);

    <memoryObjects> contains <n> names of memory objects to be deleted.
    After a memory object is deleted, it references no server memory,
    and its name is again unused.

    Unused names in <memoryObjects> are silently ignored, as is the
    value zero.

    The command

        boolean IsMemoryObjectEXT(uint memoryObject);

    returns TRUE if <memoryObject> is the name of a memory object.  If
    <memoryObject> is zero, or if <memoryObject> is a non-zero value
    that is not the name of a memory object, IsMemoryObjectEXT returns
    FALSE.

    6.1 Importing Abstract Memory into a Memory Object

    A memory object is associated with external memory by importing an
    externally-allocated abstract memory region via a reference to an
    associated external handle.  The supported set of external handle types
    and their corresponding import functions are listed in table 6.1.

        Table 6.1: Commands for importing external memory handles.

        | Handle Type | Import command |
        +-------------+----------------+
        +-------------+----------------+

    Applications must only import external memory handles exported
    from the same device or set of devices used by the current context.
    Refer to section 4.2.1 for methods to determine which devices are
    used by the current context.

    External handles are often defined using platform-specific types.
    Therefore, the base GL specification defines no methods to import an
    external handle.

    6.2 Memory object parameters

    Memory object parameters are set using the command

        void MemoryObjectParameterivEXT(uint memoryObject,
                                        enum pname,
                                        const int *params);

    <memoryObject> is the name of the memory object on which the parameter
    <pname> will be set to the value(s) in <params>.  The possible values for
    <pname> are specified in table 6.2.

        Table 6.2: Memory Object Parameters.

        | Name                        | Legal Values |
        +-----------------------------+--------------+
        | DEDICATED_MEMORY_OBJECT_EXT | FALSE, TRUE  |
        | PROTECTED_MEMORY_OBJECT_EXT | FALSE, TRUE  |
        +-----------------------------+--------------+

    The parameter DEDICATED_MEMORY_OBJECT_EXT must be set to TRUE when the
    external memory handle from which the object's memory will be imported
    was created as a dedicated allocation.

    The parameter PROTECTED_MEMORY_OBJECT_EXT must be set to TRUE when the
    external memory handle from which the object's memory will be imported
    refers to a protected resource.  The definition of a protected resource
    is outside the scope of this extension.

    Memory object parameters become immutable once the object is associated
    with external memory by an import operation.  An INVALID_OPERATION error
    is generated if <memoryObject> is immutable.

    The parameters of a memory object may be queried with the command:

        void GetMemoryObjectParameterivEXT(uint memoryObject
                                           enum pname,
                                           int *params);

    The value(s) of the parameter <pname> from the memory object
    <memoryObject> are returned in <params>.

Additions to Chapter 6 of the OpenGL 4.5 Specification (Buffer Objects)

    Modify the list of commands described in 6.2 "Creating and Modifying
    Buffer Object Data Stores" to add the following:

        void BufferStorageMemEXT(enum target,
                                 sizeiptr size,
                                 uint memory,
                                 uint64 offset);

        void NamedBufferStorageMemEXT(uint buffer,
                                      sizeiptr size,
                                      uint memory,
                                      uint64 offset);

    Replace the two paragraphs after the above list of commands with the
    following:

        "For BufferStorage and BufferStorageMemEXT, the buffer object is
        that bound to <target>, which must be one of the values listed
        in table 6.1.  For NamedBufferStorage and
        NamedBufferStorageMemEXT, <buffer> is the name of the buffer
        object.  For all the above commands, <size> is the size of the
        data store in basic machine units.  For BufferStorageMemEXT and
        NamedBufferStorageMemEXT, <memory> and <offset> define a region
        of abstract memory that will be used as the data store for
        <buffer>.  The implementation may restrict which values of
        <offset> are valid for a given memory object and buffer
        parameter combination.  These restrictions are outside the scope
        of this extension and must be determined by querying the API or
        mechanism which created the resource which <memory> refers to.
        If an invalid offset is specified an INVALID_VALUE error is
        generated.

        "The data store of the buffer object is allocated or referenced
        as a result of these commands, and cannot be de-allocated or
        unreferenced until the buffer is deleted with a call to
        DeleteBuffers."

    Replace the paragraph that beings "BufferStorage and
    NamedBufferStorage delete..." with the following:

        "BufferStorage, BufferStorageMemEXT, NamedBufferStorage, and
        NamedBufferStorageMemEXT delete any existing data store, and set
        the values of the buffer object's state variables as shown in
        table 6.3."

    Add the following to the list of errors for the BufferStorage
    functions"

        "An INVALID_VALUE error is generated by BufferStorageMemEXT and
        NamedBufferStorageMemEXT if <memory> is 0, or if <offset> +
        <size> is greater than the size of the specified
        memory object.

        "An INVALID_VALUE error is generated if <offset> is not a valid
        value for <memory> or the texture."

        "An INVALID_OPERATION error is generated if <memory> names a valid
        memory object which has no associated memory."

    Modify the header for the third column in table 6.2 to read
    "Value for *BufferStorage*", and update the table description to
    include the new memory object buffer storage commands.

    Modify the first sentence of section 6.3, "Mapping and Unmapping
    Buffer Data", to read as follows:

        "If the data store for a buffer object is not a reference to a
        memory object, all or part of the data store may be mapped into
        the client's address space with the commands:"

    Add the following to the list of errors for the MapBufferRange and
    MapNamedBufferRange commands:

        An INVALID_OPERATION error is generated by Map*BufferRange if
        the specified buffer is referencing a memory object as its data
        store.

Additions to Chapter 8 of the OpenGL 4.5 Specification (Textures and
Samplers)

    For each list of TexStorage* commands in the 1D, 2D, 3D,
    2DMultisample, and 3DMultisample families, add the following
    variants:

        void TexStorageMem*EXT(<existing parameters>,
                               uint memory,
                               uint64 offset);

        void TextureStorageMem*EXT(<existing parameters>,
                                   uint memory,
                                   uint64 offset);

    For each family of TexStorage* commands, add appropriate language to
    the description based on the following template:

        "Calling TexStorageMem*EXT or TextureStorageMem*EXT is
        equivalent to calling TexStorage* or TextureStorage*
        except that rather than allocating new memory for the texture's
        image data, the memory at <offset> in the memory object
        specified by <memory> will be used.  The implementation may
        restrict which values of <offset> are valid for a given memory
        object and texture parameter combination.  These restrictions are
        outside the scope of this extension and must be determined by
        querying the API or mechanism which created the resource which
        <memory> refers to.  If an invalid offset is specified an
        INVALID_VALUE error is generated."

    Add errors based on the following template for each family of
    TexStorage* commands:

        "An INVALID_VALUE error is generated if <memory> is 0, or if
        the memory object is not large enough to contain the specified
        texture's image data."

        "An INVALID_VALUE error is generated if <offset> is not a valid
        value for <memory> or the texture."

        "An INVALID_OPERATION error is generated if <memory> names a valid
        memory object which has no associated memory."

        "An INVALID_OPERATION error is generated if <memory> is a protected
        memory object and the texture parameter TEXTURE_PROTECTED_EXT is not
        TRUE."

    Insert the following before Table 8.17:

        "If <pname> is TEXTURE_TILING_EXT then the state is stored in the
        texture, but only takes effect the next time storage is allocated
        from a memory object for the texture object using TexStorageMem*EXT
        or TextureStorageMem*EXT.  If the value of TEXTURE_IMMUTABLE_FORMAT
        is TRUE, then TEXTURE_TILING_EXT cannot be changed and an
        INVALID_OPERATION error is generated."

    Add the following to table 8.17: Texture parameters and their values.

        | Name               | Type    | Legal values                          |
        +--------------------+---------+---------------------------------------+
        | TEXTURE_TILING_EXT | enum    | OPTIMAL_TILING_EXT, LINEAR_TILING_EXT |
        +--------------------+---------+---------------------------------------+

Additions to Chapter 22 of the OpenGL 4.5 Specification (Context state
Queries)

    Add the following to the end of the first list of functions in section
    22.1, Simple Queries:

        void GetUnsignedBytevEXT(enum pname,
                                 ubyte *data);

    Replace the sentence following that list with:

        The commands obtain boolean, integer, 64-bit integer, floating-
        point, double-precision, or unsigned byte state variables.

    Add the following to the end of the list of indexed simple state query
    commands:

        void GetUnsignedBytei_vEXT(enum target,
                                   uint index,
                                   ubyte *data);



    Add the following to section 22.3.2, Other Internal Format Queries:

        NUM_TILING_TYPES_EXT: The number of tiling types that would be
        returned by querying TILING_TYPES_EXT is returned in <params>.

        TILING_TYPES_EXT: The tiling type supported when using memory
        objects to create textures with <internalFormat> and <target>
        are written to <params>, in the order in which they occur in
        table 22.3.  Possible values are those listed in table 22.3.

        Table 22.3: Possible tiling types supported by textures using
        memory objects.

        | Tiling Type        |
        +--------------------+
        | OPTIMAL_TILING_EXT |
        | LINEAR_TILING_EXT  |
        +--------------------+

Errors

New State

Issues

    1)  Should only DSA-style texture and buffer object binding
        functions be added to keep the number of new functions
        to a minimum?

        RESOLVED: No.  Both DSA and traditional entry points will be added.

    2)  Should the type of the memory <size> and <offset> parameters be
        GLsizeiptr, GLintptr, GLint64, or GLuint64?

        RESOLVED: GLuint64.  This matches the VkDeviceSize semantics.

    3)  Should there be a way to allocate memory within OpenGL in
        addition to importing it?

        RESOLVED: No.  This could be covered in a separate extension, but
        this would involve building up all the memory property
        infrastructure Vulkan already has.  Applications wishing to use
        memory objects in OpenGL will need to leverage the allocation and
        memory capability querying mechanisms present in Vulkan to perform
        the actual allocations, and then map the capabilities to GL
        equivalents when using them.

    4)  How are sparse textures handled?

        RESOLVED: Sparse texture support is deferred to a later extension.
        Late in the development of this specification, it was discovered
        that naively extending TexPageCommitmentARB to accept an offset
        and memory object parameter results in a subtly awkward interface
        when used to build GL sparse textures equivalent to those of Vulkan
        sparse images, due to the lack of a defined memory layout ordering
        for array textures.  Developing a better interface would have
        further delayed release of the basic functionality defined here,
        which is in higher demand.

    5)  Do memory objects created as dedicated allocations need special
        handling?

        RESOLVED: No.  Like other memory regions, these allocations must be
        bound to GL objects compatible with those they are bound to in
        Vulkan to avoid aliasing issues, but otherwise no special handling
        is required.

    6)  Should the BufferStorage functions still take a flags parameter?

        RESOLVED: No.  The flags are not relevant when the memory has
        already been allocated externally.

    7)  Should the Buffer commands be called BufferStorage or BufferData?

        RESOLVED: BufferStorage.  GL has both commands, while GL ES has only
        BufferData.  The difference between the two GL commands is
        immutability.  The naming of the BufferStorage seems more consistent
        with the usage, since data is not specified with these commands, but
        a backing store is, and immutability for Vulkan memory-backed buffer
        objects seems desirable.  However, if GLES implementations can not
        support immutable buffers, BufferData() support can be added in a
        future extension with some added driver complexity.

    8)  Can semaphore commands be issued inside of Begin/End, or be
        included in display lists?

        RESOLVED: No.

    9)  Do ownership transfer and memory barrier commands need to be
        included in the semaphore operations?

        RESOLVED: Yes, these are needed for proper synchronization on some
        implementations.  Presumably only the source side of the barriers
        needs to be specified when transitioning from external to GL usage,
        and only the destination side needs to be specified when
        transitioning from GL to external usage.  That should give the
        OpenGL driver sufficient knowledge to perform any needed automatic
        transitions based on subsequent usage within the GL API.

        Still, it is unclear how much of the Vulkan pipeline barrier API
        should be explicitly exposed in the GL API:

        * Should queue ownership be included?  There is no equivalent
          idiom to define this on top of in GL.  However, since the
          external side is the only portion specified by the
          application, it could be described in Vulkan terms.

        * Should image layout be included?  Similar to the above, there
          is no GL concept of this, but Vulkan terms could be leveraged.

        * Should access type be included?  This maps relatively well to
          OpenGL memory barrier bits, but there is not a 1-1
          correspondence.

        * Should the pipeline stage be included?  This could be mapped
          to stages defined in the GL state machine, but such explicit
          references to the stages are not thus far included in GL
          language or tokens.

        Another option is to require the Vulkan driver to put images,
        buffers, and their memory in a particular state before sharing
        them with OpenGL.  For example, require applications to
        transition to the GENERAL image layout, dstStageMask of
        TOP_OF_PIPE or ALL_COMMANDS, dstAccessMask will include
        MEMORY_WRITE_BIT | MEMORY_READ_BIT or some new "more external"
        version of these, and the dstQueueFamilyIndex must be IGNORED
        while srcQueueFamilyIndex must be a valid queue family (a
        currently illegal situation).

    10) Should the barrier functionality be included in the semaphore
        operation commands?

        RESOLVED: Yes.  The only time such barriers are required in GL is
        when synchronizing with external memory accesses, which is also the
        only time semaphores should be used.  For internal synchronization,
        existing GL and EGL commands should be used.  Since the use cases
        align, it makes sense to make them a single command to cut down on
        the potential for misuse and keep the API footprint as small as
        possible.

    11) Must both Gen[MemoryObjects,Semaphores]EXT commands and
        Create[MemoryObjects,Semaphores]EXT commands be defined, or is
        one or the other sufficient?

        RESOLVED: One variant is sufficient for each object type.

    12) Should buffer objects backed by memory objects be mappable?

        RESOLVED: No.  This would complicate the API as interactions between
        GL and Vulkan cache flushing semantics would need to be defined.

    13) Does the usage information provided when creating Vulkan images
        need to be specified when creating textures on memory objects?
        If so, how is it specified?

        RESOLVED: There are a few options for specifying the usage in
        OpenGL:

        * Have some sort of GLX/EGL-like attrib list that allows users
          to specify an arbitrary set of usage parameters.

        * Allow applications to re-use the Vulkan usage flags directly
          in GL.

        * Re-define all the Vulkan image usage flags in GL, and update
          the list via new GL interop extensions as new Vulkan usage
          flags are added by Vulkan extensions.

        None of these are very compelling.  They all complicate the OpenGL
        API significantly and have a high spec maintenance burden as new
        extensions are added.

        Other options for resolving the overall issue of GL knowing the
        usage include:

        * Disallow Vulkan implementations from utilizing the usage
          information as input when determining the internal parameters of a
          Vulkan image used with external memory.

        * Only allow Vulkan implementations to utilize the usage information
          when using the dedicated allocation path where it can be stored as
          a form of metadata along with the memory.

        * Require applications to specify all supported usage flags at image
          creation time on the Vulkan side for images that are intended to
          alias with OpenGL textures.

        The first two options have the downside of potentially limiting the
        ability of implementations to fully optimize external images
        regardless of their use case.  The last option constrains the
        limitations to the case of interoperation with OpenGL, making it a
        less onerous requirement for implementations while still keeping the
        OpenGL side of the API relatively simple compared to the options
        involving re-specification of image usage on the OpenGL side.

        The agreed resolution is to use the final option: Require all
        supported usage flags be specified by the application on the Vulkan
        side if the image is intended to alias with an OpenGL texture.

    14) Are memory barriers for textures and buffer objects needed with
        semaphore signal/wait operations, or should a blanket availability/
        visibility rule be applied like in Vulkan<->Vulkan semaphore
        synchronization?

        RESOLVED: Perhaps extra availability/visibility operations need to
        be performed to enable external accesses, so it is safest to require
        explicit specification of the resources that need to be made
        available and visible as part of a semaphore synchronization
        operation.

    15) Are OpenGL equivalents of the Vulkan image creation flags related to
        sparse properties needed?

        RESOLVED: Sparse support is not included in this extension.

        Prior to this resolution, the proposed resolution was as follows:

        No.  For the purposes of OpenGL, the functionality of all the Vulkan
        sparse image creation flags is contained in the existing
        TEXTURE_SPARSE texture parameter.  Because OpenGL does not have the
        same sparse feature granularity as Vulkan, applications wishing to
        create a sparse image that will alias with an OpenGL sparse texture
        will be required to set all of the sparse bits.  Images not intended
        to alias with an OpenGL texture without the TEXTURE_SPARSE flag set
        must have none of the Vulkan sparse bits set.

    16) How do Vulkan sparse block sizes and OpenGL virtual page sizes
        interact?

        RESOLVED: Sparse support is not included in this extension.

        Prior to this resolution, the proposed resolution was as follows:

        The application must use an OpenGL virtual page size with dimensions
        matching those of the Vulkan sparse block size for any Vulkan images
        aliasing OpenGL sparse textures.  If no such virtual page size exists,
        such aliasing is not supported.

    17) Is an OpenGL equivalent of the mutable format Vulkan image creation
        parameter needed?

        RESOLVED: No.  However, Vulkan applications will be required to set
        the mutable format bit when creating an  image that will alias with
        an OpenGL texture on an OpenGL implementation that supports
        ARB_texture_view, OES_texture_view, EXT_texture_view, or OpenGL 4.3
        and above.

    18) Is an OpenGL equivalent of the tiling Vulkan image creation
        parameter needed?

        RESOLVED: Yes.  Further, OpenGL implementations may not support
        creating textures from Vulkan images using certain tiling types, so
        a query is needed to determine the types supported.

    19) Is a way to specify dedicated allocation semantics needed?

        RESOLVED: Yes.  Importing dedicated allocation-style memory may
        require the driver to use different paths than importing purely
        abstract memory.  Additionally, textures and buffer objects may need to derive meta-data from their associated memory object if
        it is a dedicated allocation.  Therefore, a dedicated allocation
        parameter should be added to the memory objects.  Additional
        parameters for textures and buffer objects are not required because
        unlike Vulkan, OpenGL exposes no application-visible texture or
        buffer state that would vary depending on whether a dedicated
        allocation will be used for their storage.  Therefore, they can
        inherit the state from the memory object associated with them at
        storage specification time.  Note that allowing parameters to be
        specified on a memory object prior to the import operation requires
        separate memory import from memory object instantiation commands.

    20) How should devices be correlated between OpenGL Vulkan, and other
        APIs?

        RESOLVED: Device UUID, LUID, and node mask queries are introduced,
        corresponding to those added to the Vulkan API for external memory/
        semaphore purposes.  Because contexts may be associated with
        multiple physical GPUs in some cases, multiple values are returned
        for device UUIDs and multiple bits are set in the device node masks.
        It is not expected that a single context will be associated with
        multiple DXGI adapters, so only one LUID is returned.

        When sharing with Vulkan device groups, the device UUIDs used by the
        context must match those of the Vulkan physical devices in the
        Vulkan device group.  Future extensions could relax this
        requirement.

    21) How do applications determine valid values for the <offset>
        parameter of the new storage allocation/binding functions?

        RESOLVED: This is outside the scope of this extension.  The API or
        mechanism which allocated the memory must provide this information.
        However, the GL will generate an error if an invalid offset is used.

    22) Are there any interactions with the EXT_protected_textures
        extension?

        RESOLVED: Yes.  Memory objects can be marked as protected or not
        protected before import.  This state must match that of the
        imported resource.  For all textures bound to a given memory object,
        the value of the TEXTURE_PROTECTED_EXT parameter of the textures
        must match the PROTECTED_MEMORY_OBJECT_EXT parameter of the memory
        object.

    23) How do applications detect when the new texture layouts
        corresponding to the image layouts in VK_KHR_maintenance2 are
        supported in OpenGL?

        RESOLVED: OpenGL contexts that report the GL_EXT_semaphore extension
        string and have a DRIVER_UUID_EXT and DEVICE_UUID_EXT corresponding
        to a Vulkan driver that supports VK_KHR_maintenance2 must support
        the new OpenGL texture layouts.

Revision History

    Revision 14, 2018-07-18 (James Jones)
        - Fixed a typo: Replace NamedBufferStroage with NamedBufferStorage

    Revision 13, 2017-09-26 (James Jones)
        - Added new image layouts corresponding to those from
          VK_KHR_maintenance2.
        - Added issue 23 and resolution.

    Revision 12, 2017-06-08 (Andres Rodriguez)
        - Fixed parameter name in MemoryObjectParameterivEXT's description.
        - Fixed missing EXT suffix in some mentions of GetUnsignedByte*

    Revision 11, 2017-06-02 (James Jones)
        - Added extension numbers.
        - Fixed the name of GetSemaphoreParameterui64vEXT.
        - Clarified which extensions each command and token belongs to.
        - Marked complete.

    Revision 10, 2017-05-24 (James Jones)
        - Added issue 21 and resolution.
        - Added issue 22 and resolution.
        - Removed sparse texture support.
        - Filled in real token values
        - Further documented the new LAYOUT tokens.

    Revision 9, 2017-04-05 (James Jones)
        - Added context device UUID queries.

    Revision 8, 2017-04-04 (James Jones)
        - Clarified semaphore semantics

    Revision 7, 2017-03-28 (James Jones)
        - Fixed various typos.

    Revision 6, 2017-03-17 (James Jones)
        - Renamed from KHR to EXT.
        - Added texture tiling parameters.
        - Added semaphore parameter manipulation functions.
        - Replaced GenMemoryObjectsEXT with CreateMemoryObjectsEXT
        - Added memory object parameter manipulation functions.
        - Updated issue 13 with a proposed resolution.
        - Added issues 15-19 and proposed resolutions.

    Revision 5, 2016-10-22 (James Jones)
        - Added proposed memory barrier semantics to the semaphore commands.
        - Added issue 14.
        - Added some clarifications to issue 13

    Revision 4, 2016-09-28 (James Jones)
        - Merged in GL_KHR_semaphore to reduce number of specs.
        - Added spec body describing the new commands.
        - Added issues 9-13.

    Revision 3, 2016-08-15 (James Jones and Jeff Juliano)
        - Clarified overview text.

    Revision 2, 2016-08-07 (James Jones)
        - Added non-contiguous sparse binding support via
          TexPageCommitmentMemKHR().

    Revision 1, 2016-08-05 (James Jones)
        - Initial draft.
