# OES_vertex_array_object

Name

    OES_vertex_array_object

Name Strings

    GL_OES_vertex_array_object

Contributors

    Ben Bowman
    Yuan Wang
    Benj Lipchak

Contact

    Ben Bowman, Imagination Technologies (benji 'dot' bowman 'at' imgtec 'dot' com)

Notice

    Copyright (c) 2009-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

IP Status


Status

    Complete. Approved by the OpenGL ES WG on November 11, 2009.
    Approved by the Khronos Board of Promoters on January 29, 2010.

Version

    Version 16, April 17, 2014

Number

    OpenGL ES Extension #71

Dependencies

    This extension is written against the OpenGL ES Common 1.1.12
    Specification (April 24, 2008) and the OpenGL ES 2.0.24
    Specification (April 22, 2009).

    OES_matrix_palette affects the definition of this extension.

Overview

    This extension introduces vertex array objects which encapsulate
    vertex array states on the server side (vertex buffer objects).
    These objects aim to keep pointers to vertex data and to provide
    names for different sets of vertex data. Therefore applications are
    allowed to rapidly switch between different sets of vertex array
    state, and to easily return to the default vertex array state.

Issues

   * Should vertex array objects be sharable across multiple OpenGL ES
     contexts?

     RESOLVED: No. The OpenGL ES working group took a straw poll and agreed
     that the advantages of compatibility with OpenGL and ease of
     implementation were more important than the disadvantage of creating
     the first non-shared named object in OpenGL ES.

   * Is it required for a vertex array object name to be explicitly
     generated before being bound?

     RESOLVED: Yes. The OpenGL ES working group agreed that
     compatibility with OpenGL and the ability to to guide developers to
     more "future looking" object usage were more important than keeping
     consistency with other objects in OpenGL ES 1 and 2.

   * Should a vertex array object be allowed to encapsulate client
     vertex arrays?

     RESOLVED: No. The OpenGL ES working group agreed that compatibility
     with OpenGL and the ability to to guide developers to more
     performant drawing by enforcing VBO usage were more important than
     the possibility of hurting adoption of VAOs.

   * Should client array indices be employed by DrawElements when a
     non-zero vertex array object is bound?

     RESOLVED: Yes. The original ARB_vao and OpenGL 3.0 incarnations of
     this feature allowed client index data, so this extension should
     also.

   * When an application attempts to utilise a zero-named vertex array
     buffer or a zero-named element array buffer, while a non-zero
     vertex array object is presently attached, what should happen?

     RESOLVED: Generally speaking, these kinds of endeavours are
     erroneous, but some cases are deliberately tolerated. And they are
     detailed as follows:

     - Binding a zero-named vertex array buffer:
       this can be detected by *Pointer(ES1) or VertexAttribPointer(ES2);

       - if the pointer argument is not NULL:
         this means to bind a client vertex array;
         an INVALID_OPERATION error will be returned.

       - if the pointer argument is NULL:
         the result or drawing will be undefined, but with no GL error;
         this enables a previously encapsulated vertex array buffer to
         be detached from a vertex array object.

     - Binding a zero-named element array buffer:
       this can be identified by DrawElements;

       - no restrictions.


New Procedures and Functions

    void BindVertexArrayOES(uint array);

    void DeleteVertexArraysOES(sizei n, const uint *arrays);

    void GenVertexArraysOES(sizei n, uint *arrays);

    boolean IsVertexArrayOES(uint array);

New Tokens

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, GetFloatv:

        VERTEX_ARRAY_BINDING_OES    0x85B5

Additions to Chapter 2 (OpenGL ES Operation)

   * Add a new Section "2.X Vertex Array Objects" after Section "2.9
     Buffer Objects".

     The buffer objects that are to be used by the vertex stage of the
     GL ES are collected together to form a vertex array object. All
     state related to the definition of the data used by the vertex
     processor is encapsulated in a vertex array object.

     The command

          void GenVertexArraysOES(sizei n, uint *arrays);

     returns <n> previous unused vertex array object names in <arrays>.
     These names are marked as used, for the purposes of
     GenVertexArraysOES only, but they acquire array state only when
     they are first bound, just as if they were unused.

     Vertex array objects are deleted by calling

          void DeleteVertexArraysOES(sizei n, const uint *arrays);

     <arrays> contains <n> names of vertex array objects to be deleted.
     Once a vertex array object is deleted it has no contents and its
     name becomes unused again. If a vertex array object that is
     currently bound is deleted, the binding for that object reverts to
     zero and the default vertex array becomes current. Unused names in
     <arrays> are silently ignored, as if they have the value zero.

     The name space for vertex array objects is the unsigned integers,
     with zero reserved for the GL ES.

     A vertex array object is created by binding an unused name with the
     command

          void  BindVertexArrayOES(uint array);

IF(ES1)

     and <array> here is the vertex array object name. The resulting
     vertex array object is a new state vector, comprising all the state
     values (listed in Tables 6.4 & 6.5, except CLIENT_ACTIVE_TEXTURE
     and ARRAY_BUFFER_BINDING):

     VERTEX_ARRAY,
     VERTEX_ARRAY_SIZE,
     VERTEX_ARRAY_TYPE,
     VERTEX_ARRAY_STRIDE,
     VERTEX_ARRAY_POINTER,
     VERTEX_ARRAY_BUFFER_BINDING,

     NORMAL_ARRAY,
     NORMAL_ARRAY_TYPE,
     NORMAL_ARRAY_STRIDE,
     NORMAL_ARRAY_POINTER,
     NORMAL_ARRAY_BUFFER_BINDING,

     COLOR_ARRAY,
     COLOR_ARRAY_SIZE,
     COLOR_ARRAY_TYPE,
     COLOR_ARRAY_STRIDE,
     COLOR_ARRAY_POINTER,
     COLOR_ARRAY_BUFFER_BINDING,

     TEXTURE_COORD_ARRAY,
     TEXTURE_COORD_ARRAY_SIZE,
     TEXTURE_COORD_ARRAY_TYPE,
     TEXTURE_COORD_ARRAY_STRIDE,
     TEXTURE_COORD_ARRAY_POINTER,
     TEXTURE_COORD_ARRAY_BUFFER_BINDING,

     POINT_SIZE_ARRAY_OES,
     POINT_SIZE_ARRAY_TYPE_OES,
     POINT_SIZE_ARRAY_STRIDE_OES,
     POINT_SIZE_ARRAY_POINTER_OES,
     POINT_SIZE_ARRAY_BUFFER_BINDING_OES,

     ELEMENT_ARRAY_BUFFER_BINDING.

     BindVertexArrayOES may also be used to bind an existing vertex
     array object. If the binding is successful, no change is made to
     the state of the bound vertex array object, and any previous
     binding is broken.

     BindVertexArrayOES fails and an INVALID_OPERATION error is
     generated if array is not a name returned from a previous call to
     GenVertexArraysOES, or if such a name has since been deleted with
     DeleteVertexArraysOES. An INVALID_OPERATION error is generated if
     any of the *Pointer commands specifying the location and
     organization of vertex array data are called while a non-zero
     vertex array object is bound, zero is bound to the ARRAY_BUFFER
     buffer object binding point, and the pointer argument is not NULL
     [fn1].
        [fn1: This error makes it impossible to create a vertex array
         object containing client array pointers, while still allowing
         buffer objects to be unbound.]

     The currently bound vertex array object is used for all commands
     that modify vertex array state, such as
         VertexPointer,
         NormalPointer,
         ColorPointer,
         TexCoordPointer,
         PointSizePointerOES,
         EnableClientState,
         DisableClientState,
         ClientActiveTexture,
     all commands that draw from vertex arrays, such as
         DrawArrays
         DrawElements,
     and all queries of vertex array state (see Chapter 6).

END(ES1)

IF(ES2)

     and <array> here is the vertex array object name. The resulting
     vertex array object is a new state vector, comprising all the state
     values (listed in Table 6.2, except ARRAY_BUFFER_BINDING):

          VERTEX_ATTRIB_ARRAY_ENABLED,
          VERTEX_ATTRIB_ARRAY_SIZE,
          VERTEX_ATTRIB_ARRAY_STRIDE,
          VERTEX_ATTRIB_ARRAY_TYPE,
          VERTEX_ATTRIB_ARRAY_NORMALIZED,
          VERTEX_ATTRIB_ARRAY_POINTER,
          ELEMENT_ARRAY_BUFFER_BINDING,
          VERTEX_ATTRIB_ARRAY_BUFFER_BINDING.

     BindVertexArrayOES may also be used to bind an existing vertex
     array object. If the binding is successful, no change is made to
     the state of the bound vertex array object, and any previous
     binding is broken.

     BindVertexArrayOES fails and an INVALID_OPERATION error is
     generated if array is not a name returned from a previous call to
     GenVertexArraysOES, or if such a name has since been deleted with
     DeleteVertexArraysOES. An INVALID_OPERATION error is generated if
     VertexAttribPointer is called while a non-zero vertex array object
     is bound, zero is bound to the <ARRAY_BUFFER> buffer object binding
     point and the pointer argument is not NULL [fn1].
        [fn1: This error makes it impossible to create a vertex array
         object containing client array pointers, while still allowing
         buffer objects to be unbound.]

     The currently bound vertex array object is used for all commands
     that modify vertex array state, such as VertexAttribPointer and
     EnableVertexAttribArray; all commands that draw from vertex arrays,
     such as DrawArrays and DrawElements; and all queries of vertex
     array state (see Chapter 6).

END(ES2)

     And the presently attached vertex array object has the following
     impacts on the draw commands:

          While a non-zero vertex array object is bound, if any enabled
          array's buffer binding is zero, when DrawArrays or
          DrawElements is called, the result is undefined.

Additions to Chapter 3 (Rasterization)

   * None.


Additions to Chapter 4 (Per-Fragment Operations and the Framebuffer)

   * None.


Additions to Chapter 5 (Special Functions)

   * None.


Additions to Chapter 6 (State and State Requests)

IF(ES1)

   * Add a new paragraph at the end of Section 6.1.2: Data Conversions
     (Page 119).

     Vertex array state variables are qualified by the value of
     VERTEX_ARRAY_BINDING_OES to determine which vertex array object is
     queried. Tables 6.4 & 6.5 define the set of state stored in a
     vertex array object.

   * Add a new Section "6.1.X Vertex Array Object Queries" after
     Sections "6.1.6 Buffer Object Queries" (Page 120).

     The command

          boolean IsVertexArrayOES(uint array);

     returns TRUE if <array> is the name of a vertex array object. If
     <array> is zero, or a non-zero value that is not the name of a
     vertex array object, IsVertexArrayOES returns FALSE. No error is
     generated if <array> is not a valid array object name.

END(ES1)

IF(ES2)

   * Add a new paragraph at the end of Section 6.1.2: Data Conversions
     (Page 123).

     Vertex array state variables are qualified by the value of
     VERTEX_ARRAY_BINDING_OES to determine which vertex array object is
     queried. Table 6.2 defines the set of state stored in a vertex
     array object.

   * Add a new Section "6.1.X Vertex Array Object Queries" between
     Sections "6.1.6 Buffer Object Queries" and "6.1.7 Framebuffer
     Object and Renderbuffer Queries" (Page 126).

     The command

          boolean IsVertexArrayOES(uint array);

     returns TRUE if <array> is the name of a vertex array object. If
     <array> is zero, or a non-zero value that is not the name of a
     vertex array object, IsVertexArray returns FALSE. No error is
     generated if <array> is not a valid array object name.

   * Modify lines in Section 6.1.8: Shader and Program Queries

     Page 131: replace "Note that all the queries except
     CURRENT_VERTEX_ATTRIB return client state."
        with
     "Note that all the queries except CURRENT_VERTEX_ATTRIB return
     values stored in the currently bound vertex array object (the value
     of VERTEX_ARRAY_BINDING). If the zero object is bound, then the
     queries return client state."

     Page 131: add lines after "<pname> must be VERTEX_ATTRIB_ARRAY_POINTER."
     The value returned is queried from the currently bound vertex array
     object. If the zero object is bound, the value is queried from
     client state.


Additions to Appendix C (Deleting Shared Objects)

   * Add another section in C.2 Sharing objects across multiple OpenGL
     ES contexts (Page 163).

     Objects which cannot be shared in this manner include:
          vertex array objects

END(ES2)

IF(ES1)

Dependencies on OES_matrix_palette

    If OES_matrix_palette is supported, the language below should be
    added into Chapter 2.

    The vertex array object can also comprise the following state values:
        MATRIX_INDEX_ARRAY_OES,
        MATRIX_INDEX_ARRAY_SIZE_OES,
        MATRIX_INDEX_ARRAY_TYPE_OES,
        MATRIX_INDEX_ARRAY_STRIDE_OES,
        MATRIX_INDEX_ARRAY_POINTER_OES,
        MATRIX_INDEX_ARRAY_BUFFER_BINDING_OES,

        WEIGHT_ARRAY_OES,
        WEIGHT_ARRAY_SIZE_OES,
        WEIGHT_ARRAY_TYPE_OES,
        WEIGHT_ARRAY_STRIDE_OES,
        WEIGHT_ARRAY_POINTER_OES,
        WEIGHT_ARRAY_BUFFER_BINDING_OES.

    And the currently bound vertex array object can also be used for the
    following commands
        WeightPointerOES,
        MatrixIndexPointerOES.

END(ES1)

Revision History

     Rev.      Date        Author                    Changes
   -------  ----------  ------------  -----------------------------------------------------
     16      04/17/14     Jon Leech    Update wording of first issue (Bug 7847).
     15      02/05/10     Jon Leech    Update Status and assign extension number.
                                       Reflow spec text to standard widths.
     14      10/11/09     Ben Bowman   Allow VAO to employ client indices.
     13      09/11/09     Yuan Wang    Forbade VAO to employ client array indices.
     12      04/11/09     Ben Bowman   Updated EXT to OES and adopted resolutions of the
                                       OpenGL ES working group meeting on 04/11/09. (re)
                                       disallow user generated (non Gen'd) names.
     11      23/10/09     Ben Bowman   Fixed INVALID_OPERATION error for client arrays.
     10      14/10/09     Ben Bowman   Adopted resolutions of the OpenGL ES working group
                                       meeting on 14/10/09. (re) disallow client arrays to
                                       be included in VAOs. Disallow sharing of VAOs.
     9       14/10/09     Yuan Wang    Added dependencies on OES_matrix_palette
     8       14/10/09     Ben Bowman   Formatting. Reopen of shared object decision.
     7       09/09/09     Ben Bowman   Cleanup of issues list.
     6       04/09/09     Yuan Wang    Changed contact details and added contributors.
                                       Replaced IMG with EXT, added the extension for ES1
                                       specification and added revision history.
     5       20/08/09     Yuan Wang    Allowed VAO to encapsulate client attribute arrays.
     4       16/07/09     Yuan Wang    Added ELEMENT_ARRAY_BUFFER_BINDING into VAO's
                                       contained states, but took ARRAY_BUFFER_BINDING
                                       out of the states.
     3       07/07/09     Yuan Wang    Removed VERTEX_ARRAY_IMG token, and removed target
                                       param from BindVertexArrayIMG, since there is only
                                       one type of target for VAO.
     2       26/06/09     Yuan Wang    Added IMG affix and VERTEX_ARRAY target for
                                       BindVertexArrayIMG.
     1       15/06/09     Yuan Wang    Initial revision.




