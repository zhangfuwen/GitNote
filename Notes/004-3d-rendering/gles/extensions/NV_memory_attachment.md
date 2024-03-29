# NV_memory_attachment

Name

    NV_memory_attachment

Name Strings

    GL_NV_memory_attachment

Contributors

    Carsten Rohde, NVIDIA
    Christoph Kubisch, NVIDIA
    James Jones, NVIDIA

Contact

    Carsten Rohde, NVIDIA (crohde 'at' nvidia.com)

Status

    Complete

Version

    Last Modified Date: Aug 27, 2018
    Revision: 2

Number

    524
    OpenGL ES Extension #305

Dependencies

    Requires GL_EXT_memory_object and ARB_texture_storage or a version of
    OpenGL or OpenGL ES that incorporates it.

    Written against the OpenGL 4.6 and OpenGL ES 3.2 specifications.

    Interacts with ARB_direct_state_access (OpenGL) when OpenGL < 4.5 is used.

    Interacts with NV_shader_buffer_load.

    Interacts with NV_bindless_texture and ARB_bindless_texture.

Overview

    This extension extends the memory objects introduced with EXT_memory_object
    to allow existing textures and buffers to be migrated to an imported memory
    allocation.  The primary use-case of this extension is plug-in development
    where resource management (creation, deletion, sizing etc.) is handled by
    inaccessible host application code.

New Procedures and Functions

    If the GL_NV_memory_attachment string is reported, the following
    commands are added:

        void GetMemoryObjectDetachedResourcesuivNV(uint memory,
                                                   enum pname,
                                                   int first,
                                                   sizei count,
                                                   uint *params)

        void ResetMemoryObjectParameterNV(uint memory,
                                          enum pname);

        void TexAttachMemoryNV(enum target,
                               uint memory,
                               uint64 offset);

        void BufferAttachMemoryNV(enum target,
                                  uint memory,
                                  uint64 offset);

        [[ The following are added if direct state access is supported ]]

        void TextureAttachMemoryNV(uint texture,
                                   uint memory,
                                   uint64 offset);

        void NamedBufferAttachMemoryNV(uint buffer,
                                       uint memory,
                                       uint64 offset);

New Tokens

    If the GL_NV_memory_attachment string is reported, the following tokens
    are added:

    Accepted by the <pname> parameter of TexParameter{ifx}{v},
    TexParameterI{i ui}v, TextureParameter{if}{v}, TextureParameterI{i ui}v,
    GetTexParameter{if}v, GetTexParameterI{i ui}v, GetTextureParameter{if}v,
    GetTextureParameterI{i ui}v, GetBufferParameter{i|i64}v and
    GetNamedBufferParameter{i|i64}v:

      ATTACHED_MEMORY_OBJECT_NV           0x95A4
      ATTACHED_MEMORY_OFFSET_NV           0x95A5
      MEMORY_ATTACHABLE_ALIGNMENT_NV      0x95A6
      MEMORY_ATTACHABLE_SIZE_NV           0x95A7
      MEMORY_ATTACHABLE_NV                0x95A8

    Accepted by the <pname> parameter of GetBooleanv, GetDoublev, GetFloatv,
    GetIntegerv, GetInteger64v, GetUnsignedBytevEXT,
    GetMemoryObjectParameterivEXT, and the <target> parameter of GetBooleani_v,
    GetIntegeri_v,GetFloati_v, GetDoublei_v, GetInteger64i_v and
    GetUnsignedBytei_vEXT:

      DETACHED_MEMORY_INCARNATION_NV      0x95A9

    Accepted by the <pname> parameter of GetMemoryObjectParameterivEXT,
    GetMemoryObjectDetachedResourcesuivNV and ResetMemoryObjectParameterNV:

      DETACHED_TEXTURES_NV                0x95AA
      DETACHED_BUFFERS_NV                 0x95AB

    Accepted by the <pname> parameter of MemoryObjectParameterivEXT,
    GetMemoryObjectParameterivEXT:

      MAX_DETACHED_TEXTURES_NV            0x95AC
      MAX_DETACHED_BUFFERS_NV             0x95AD


Additions to Chapter 6 of the EXT_external_objects Specification
(Memory Objects)

    Add a new sections after 6.2 (Memory object parameters)

        6.3 Attaching memory to existing resources

        MEMORY_ATTACHABLE_NV should be used to query if it is valid to attach
        a memory object to an existing resource (buffer or texture).  The
        memory region size and offset alignment required by a resource can be
        queried using MEMORY_ATTACHABLE_SIZE_NV and
        MEMORY_ATTACHABLE_ALIGNMENT_NV respectively.  The current attached
        memory object and the used offset for a resource can be queried by
        ATTACHED_MEMORY_OBJECT_NV and ATTACHED_MEMORY_OFFSET_NV.

        If a resource which has memory attached is resized, the attached memory
        will be detached and a new data store will be allocated.  If a resource
        which has memory attached is deleted, the attached memory will first be
        detached.  If any such detachment occurs, a global incarnation counter
        will be increased and the current value will be stored in the detached
        memory object.  The incarnation counter can be queried by
        DETACHED_MEMORY_INCARNATION_EXT either globally or for a specific
        memory object.

        The command

            void GetMemoryObjectDetachedResourcesuivNV(uint memory,
                                                       enum pname,
                                                       int first,
                                                       sizei count,
                                                       uint *params)

        will return a list of detached buffers (if <pname> is
        DETACHED_BUFFERS_NV) or textures (if <pname> is DETACHED_TEXTURES_NV)
        in <params> for memory object <memory>.  It will return <count> items
        beginning with <first> item.  The number of available items can be
        queried by calling GetMemoryObjectParameterivEXT with <pname> set to
        DETACHED_TEXTURES_NV or DETACHED_BUFFERS_NV.  An INVALID_VALUE error is
        generated by GetMemoryObjectDetachedResourcesuivNV if <memory> is 0.
        An INVALID_OPERATION error is generated if <memory> names a valid
        memory object which has no associated memory.  An INVALID_VALUE error
        is generated if <pname> is neither DETACHED_BUFFERS_NV nor
        DETACHED_TEXTURES_NV.  An INVALID_VALUE error is generated if
        <first> + <count> is greater than the number of available items in the
        list.  An INVALID_VALUE error is generated if <params> is NULL.
        MemoryObjectParameterivEXT must be called with <pname> set to
        MAX_DETACHED_TEXTURES_NV or MAX_DETACHED_BUFFERS_NV before calling
        GetMemoryObjectDetachedResourcesuivNV to set the maximum number of
        items in the list of detached textures or buffers.  The default values
        are 0 which means that tracking of detached textures and buffers is
        disabled by default.

        The command

        void ResetMemoryObjectParameterNV(uint memory,
                                          enum pname);

        will reset the list of detached buffers (if <pname> is
        DETACHED_BUFFERS_NV) or textures (if <pname> is DETACHED_TEXTURES_NV)
        for memory object <memory>.  An INVALID_VALUE error is generated by
        ResetMemoryObjectParameterNV if <memory> is 0.  An INVALID_OPERATION
        error is generated if <memory> names a valid memory object which has
        no associated memory.  An INVALID_VALUE error is generated if <pname>
        is neither DETACHED_BUFFERS_NV nor DETACHED_TEXTURES_NV.


Additions to Chapter 6 of the OpenGL 4.6 Specification (Buffer Objects)

    Add a new section after 6.2.1 (Clearing Buffer Object Data Stores)

        6.2.2 Attaching a memory object to a buffer object

        The commands

            void BufferAttachMemoryNV(enum target,
                                      uint memory,
                                      uint64 offset);

            void NamedBufferAttachMemoryNV(uint buffer,
                                           uint memory,
                                           uint64 offset);

        will attach a region of a memory object to a buffer object.  For
        BufferAttachMemoryNV, the buffer object is that bound to <target>,
        which must be one of the values listed in table 6.1.  For
        NamedBufferAttachMemoryNV, <buffer> is the name of the buffer
        object.  <memory> and <offset> define a region of memory that will
        replace the data store for <buffer>. The content of the original data
        store will be preserved by a server side copy and the original data
        store will be deleted after that copy.  The implementation may restrict
        which values of <offset> are valid for a given memory object and buffer
        parameter combination.  These restrictions are outside the scope of
        this extension and must be determined by querying the API or mechanism
        which created the resource which <memory> refers to.  If an invalid
        offset is specified an INVALID_VALUE error is generated.  An
        INVALID_VALUE error is generated by BufferAttachMemoryNV and
        NamedBufferAttachMemoryNV if <memory> is 0. An INVALID_OPERATION error
        is generated if <memory> names a valid memory object which has no
        associated memory.  An INVALID_OPERATION error is generated if the
        specified buffer was created with MAP_PERSISTENT_BIT flag.  An
        INVALID_OPERATION error is generated if the specified buffer is
        currently mapped by client.

Additions to Chapter 8 of the OpenGL 4.6 Specification (Textures and
Samplers)

    Add a new section between sections 8.19, "Immutable-Format Texture Images"
    and section 8.20, "Invalidating Texture Image Data"

        8.20 Attaching a memory object to a texture image

        The commands

            void TexAttachMemoryNV(enum target,
                                   uint memory,
                                   uint64 offset);

            void TextureAttachMemoryNV(uint texture,
                                       uint memory,
                                       uint64 offset);

        will attach a region of a memory object to a texture.  For
        TexAttachMemoryNV, the texture is that bound to <target>, which must be
        one of TEXTURE_1D, TEXTURE_2D, TEXTURE_3D, TEXTURE_1D_ARRAY,
        TEXTURE_2D_ARRAY, TEXTURE_RECTANGLE, TEXTURE_CUBE_MAP,
        TEXTURE_CUBE_MAP_ARRAY, TEXTURE_2D_MULTISAMPLE, or
        TEXTURE_2D_MULTISAMPLE_ARRAY.  For TextureAttachMemoryNV, <texture> is
        the name of the texture.  <memory> and <offset> define a region of
        memory that will replace the data store for <texture>. The content of
        the original data store will be preserved by a server side copy and the
        original data store will be deleted after that copy.  The
        implementation may restrict which values of <offset> are valid for a
        given memory object and texture parameter combination.  These
        restrictions are outside the scope of this extension and must be
        determined by querying the API or mechanism which created the resource
        which <memory> refers to.  If an invalid offset is specified an
        INVALID_VALUE error is generated.  An INVALID_VALUE error is generated
        by TexAttachMemoryNV and TextureAttachMemoryNV if <memory> is 0.  An
        INVALID_OPERATION error is generated if <memory> names a valid memory
        object which has no associated memory.

Errors

New State

Sample

    // host: code not visible to the plug-in developer
    // plug-in: code written by plug-in developer

    uint tex0;
    uint tex1;

    // host
    {
        // sets up textures as usual
    }

    // plug-in
    {
        int attachable0;
        int attachable1;
        GetTextureParameteriv(tex0, MEMORY_ATTACHABLE_NV, &attachable0);
        GetTextureParameteriv(tex1, MEMORY_ATTACHABLE_NV, &attachable1);

        if (attachable0 && attachable1){

            // allocate memory within vulkan and import it as specified in
            // EXT_memory_object

            // attach imported vulkan memory
            TextureAttachMemoryNV(tex0, memobj, memoffset0);

            // ... do same for tex1
            TextureAttachMemoryNV(tex1, memobj, memoffset1);
        }
    }

    ///////////////////////////////
    // Querying mutations by host

    int incarnationExpected;

    // plug-in
    {
        // global query
        GetIntegerv(DETACHED_MEMORY_INCARNATION_NV, &incarnationExpected);

        // if we have multiple memory objects
        for each memobj {
          GetMemoryObjectParameterivEXT(memobj.id,
                                        DETACHED_MEMORY_INCARNATION_NV,
                                        &memobj.incarnation);
          GLint maxDetachedTextures = 64;
          MemoryObjectParameterivEXT(memobj.id,
                                     MAX_DETACHED_TEXTURES_NV,
                                     &maxDetachedTextures);
        }
    }

    // host
    {
        // deletion triggers a detach
        glDeleteTextures(1, &tex1);
    }

    // plug-in
    {
        // global query if resources of context were affected
        int incarnation;
        GetIntegerv(DETACHED_MEMORY_INCARNATION_NV, &incarnation);

        if (incarnation != incarnationExpected) {
            incarnationExpected = incarnation;

            // narrow down search which memory object was affected
            for each memobj {
                GetMemoryObjectParameterivEXT(memobj.id,
                                              DETACHED_MEMORY_INCARNATION_NV,
                                              &incarnation);

                if (incarnation != memobj.incarnation) {
                    memobj.incarnation = incarnation;

                    int removedTexCount;
                    GetMemoryObjectParameterivEXT(memobj.id,
                                                  DETACHED_TEXTURES_NV,
                                                  &removedTexCount);

                    std::vector<uint> removedTexs(removedTexCount);

                    GetMemoryObjectDetachedResourcesuivNV(
                        memobj.id,
                        DETACHED_TEXTURES_NV,
                        0, removedTexCount,
                        removedTexs.data());

                    for (int i = 0; i < removedTexCount; i++) {
                        uint tex = removedTexs[i];
                        // look up tex in custom allocator and
                        // mark its memory as available again
                    }

                    ResetMemoryObjectParameter(memobj.id,
                                               DETACHED_TEXTURES_NV);
                }
            }
        }
    }

Issues

    1)  Do we need to introduce allocation done within OpenGL
        or is attaching existing resources to imported allocation
        sufficient?

        RESOLVED: No.  No need to duplicate work which has already been done
        in Vulkan.

    2)  Should binding memory only work on immutable resources?

        RESOLVED: No.  To improve compatibility with existing GL resources,
        allow mutable resources as well. A global and local incarnation counter
        was introduced to test against changes, as well as detecting the
        detached resources.

    3)  Do we support client-mappable resources?

        RESOLVED: Yes.  Client-mappable resources are supported but not
        when they are persistent. When memory is attached resource must be
        unmapped.

    4)  What are the affects on TextureViews?

        RESOLVED: TextureViews inherit the memory state.

    5)  Do bindless resources change?

        RESOLVED: Yes.  The existing handles and GPU addresses become invalid
        when memory is attached and must be queried afterwards.

    6)  Should we support resources that were migrated to host memory by driver?

        RESOLVED: Yes, but the attached memory is independ from this state.

    7)  Do we need an "attachable" per-resource state?

        RESOLVED: Yes.

    8)  How is bindless residency affected?

        RESOLVED: A memory object becomes resident if at least one attached
        resource is resident.


Revision History

    Revision 2, 2018-08-20 (Carsten Rohde, Christoph Kubisch)
        - Added spec body describing new commands.
        - Added non-DSA functions
        - Resolve issues

    Revision 1, 2018-05-07 (Carsten Rohde, Christoph Kubisch)
        - Initial draft.
