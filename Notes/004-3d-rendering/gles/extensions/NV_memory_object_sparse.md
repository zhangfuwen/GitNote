# NV_memory_object_sparse

Name

    NV_memory_object_sparse

Name Strings

    GL_NV_memory_object_sparse

Contributors

    Carsten Rohde, NVIDIA
    James Jones, NVIDIA

Contact

    Carsten Rohde, NVIDIA Corporation (crohde 'at' nvidia.com)

Status

    Complete

Version

    Last Modified Date: August 14, 2020
    Revision:           2

Number

    550
    OpenGL ES Extension #329

Dependencies

    Written against the OpenGL 4.6 and OpenGL ES 3.2 specifications
    including ARB_sparse_texture and ARB_sparse_buffer.

    GL_NV_memory_object_sparse requires GL_EXT_memory_object,
    ARB_sparse_texture and/or ARB_sparse_buffer or a version of
    OpenGL or OpenGL ES that incorporates it.

    NV_memory_object_sparse interacts with ARB_direct_state_access (OpenGL)
    when OpenGL < 4.6 is used.

    ARB_sparse_texture (OpenGL) interacts with GL_EXT_memory_object_sparse.
    ARB_sparse_buffer (OpenGL) interacts with GL_EXT_memory_object_sparse.
    EXT_sparse_texture (OpenGL ES) interacts with GL_EXT_memory_object_sparse.

Overview

    This extension adds sparse support to EXT_memory_object extension.

New Procedures and Functions

    void BufferPageCommitmentMemNV(enum target,
                                   intptr offset, sizeiptr size,
                                   uint memory, uint64 memOffset,
                                   boolean commit);

    void NamedBufferPageCommitmentMemNV(uint buffer,
                                        intptr offset, sizeiptr size,
                                        uint memory, uint64 memOffset,
                                        boolean commit);

    void TexPageCommitmentMemNV(enum target,
                                int layer, int level,
                                int xoffset, int yoffset, int zoffset,
                                sizei width, sizei height, sizei depth,
                                uint memory, uint64 offset,
                                boolean commit);

    void TexturePageCommitmentMemNV(uint texture,
                                    int layer, int level,
                                    int xoffset, int yoffset, int zoffset,
                                    sizei width, sizei height, sizei depth,
                                    uint memory, uint64 offset,
                                    boolean commit);


New Tokens

    None


Additions to Chapter 6 of the OpenGL 4.4 (core) Specification (Buffer Objects)

    In Section 6.2, "Creating and Modifying Buffer Object Data Stores", add
    the following add the following to the end of the description of
    BufferSubData:

    If NV_memory_object_sparse is supported, additionally, the commands

        void BufferPageCommitmentMemNV(enum target,
                                       intptr offset, sizeiptr size,
                                       uint memory, uint64 memOffset,
                                       boolean commit);

        void NamedBufferPageCommitmentMemNV(uint buffer,
                                            intptr offset,
                                            sizeiptr size,
                                            uint memory, uint64 memOffset,
                                            boolean commit);

    behaves similarly to BufferPageCommitmentARB and
    NamedBufferPageCommitmentMemARB except that the pages of the sparse buffer
    are bound to the memory specified by <memory> and <memOffset>.

    Errors (additionally to non-Mem variants)

      An INVALID_OPERATION error is generated if <memory> is not the name of
      an existing memory object.

      An INVALID_OPERATION error is generated if <offset> + <size> exceeds the
      size of the memory object.

    Add the following to end of subsection 8.20.2. "Controlling Sparse Texture
    Commitment":

    If NV_memory_object_sparse is supported, additionally, the commands

        void TexPageCommitmentMemNV(enum target,
                                    int layer, int level,
                                    int xoffset, int yoffset, int zoffset,
                                    sizei width, sizei height, sizei depth,
                                    uint memory, uint64 offset,
                                    boolean commit);

        void TexturePageCommitmentMemNV(uint texture,
                                        int layer, int level,
                                        int xoffset, int yoffset, int zoffset,
                                        sizei width, sizei height, sizei depth,
                                        uint memory, uint64 offset,
                                        boolean commit);

    behaves similarly to TexPageCommitmentMemARB and
    TexturePageCommitmentEXT except that the tiles of the sparse texture
    are bound to the memory specified by <memory> and <offset>.

    <layer> indicates the layer of a texture array or cube texture,
    <zoffset> must be 0 and <depth> must 1 in this case. For other textures
    <layer> must be 0.

    Errors (additionally to non-Mem variants)

      An INVALID_OPERATION error is generated if <memory> is not the name of
      an existing memory object.

      An INVALID_OPERATION error is generated if <memory> is dedicated or
      imported from a non-opaque handle.

      An INVALID_OPERATION error is generated if <offset> plus the number of
      bytes required for the tiles to commit exceeds the size of the memory
      object.

      An INVALID_VALUE error is generated if <layer> is not 0 and the texture
      neither a texture array or a cube texture.

      An INVALID_VALUE error is generated if <zoffset> is not 0 or <depth> is
      not 1 if the texture is a texture array or a cube texture.

      An INVALID_VALUE error is generated if <layer> is greater or equal than
      the number of layers of the texture array.

      An INVALID_VALUE error is generated if <layer> is greater or equal than
      6 in case of a cube texture.

Dependencies on EXT_direct_state_access

    If EXT_direct_state_access is not supported, remove references to the
    NamedBufferPageCommitmentMemNV and TexturePageCommitmentMemNV commands
    added by this extension.

Issues

    (1) Should we a 'aspect' parameter to the new gl.*CommitMemNV() functions?

    RESOLVED: No. This can be deferred to a future EXT extension because there
              is currently no multi-planar texture support in GL and metadata
              isn't required for NVIDIA hardware.

Revision History

    Revision 2, 2020-08-14 (Piers Daniell)
        - Fix duplicate parameter names in BufferPageCommitmentMemNV and
          NamedBufferPageCommitmentMemNV.

    Revision 1, 2020-08-04 (Carsten Rohde)
        - Initial draft.
