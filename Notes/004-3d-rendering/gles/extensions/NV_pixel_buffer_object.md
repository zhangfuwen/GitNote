# NV_pixel_buffer_object

Name

    NV_pixel_buffer_object

Name Strings

    GL_NV_pixel_buffer_object

Contributors

    Contributors to ARB_pixel_buffer_object
    Greg Roth, NVIDIA

Contact

    Mathias Heyer, NVIDIA Corporation (mheyer 'at' nvidia.com) 

Status

    Complete.

Version

    Last Modified Date: April 27th, 2020
    Revision: 3.0

Number

    OpenGL ES Extension #134

Dependencies

    Written based on the wording of the OpenGL ES 2.0 specification.
    
    OES_mapbuffer affects the definition of this specification
    EXT_map_buffer_range affects the definition of this specification 

Overview

    This extension permits buffer objects to be used not only with vertex
    array data, but also with pixel data.  The intent is to provide more
    acceleration opportunities for OpenGL pixel commands.

    While a single buffer object can be bound for both vertex arrays and
    pixel commands, we use the designations vertex buffer object (VBO)
    and pixel buffer object (PBO) to indicate their particular usage in
    a given situation.

    This extension does not add any new functionality to buffer objects
    themselves.  It simply adds two new targets to which buffer objects
    can be bound: GL_PIXEL_PACK_BUFFER_NV and GL_PIXEL_UNPACK_BUFFER_NV.
    When a buffer object is bound to the GL_PIXEL_PACK_BUFFER_NV target,
    commands such as glReadPixels pack (write) their data into a buffer
    object. When a buffer object is bound to the GL_PIXEL_UNPACK_BUFFER_NV
    target, commands such as glTexImage2D unpack (read) their
    data from a buffer object.

    There are a several approaches to improve graphics performance
    with PBOs.  Some of the most interesting approaches are:

    - Streaming texture updates:  If the application uses
      glMapBufferOES/glMapBufferRangeEXT/glUnmapBufferOES to write
      its data for glTexSubImage into a buffer object, at least one of
      the data copies usually required to download a texture can be
      eliminated, significantly increasing texture download performance.

    - Asynchronous glReadPixels:  If an application needs to read back a
      number of images and process them with the CPU, the existing GL
      interface makes it nearly impossible to pipeline this operation.
      The driver will typically send the hardware a readback command
      when glReadPixels is called, and then wait for all of the data to
      be available before returning control to the application.  Then,
      the application can either process the data immediately or call
      glReadPixels again; in neither case will the readback overlap with
      the processing.  If the application issues several readbacks
      into several buffer objects, however, and then maps each one to
      process its data, then the readbacks can proceed in parallel with
      the data processing.

    - Render to vertex array:  The application can use a fragment
      program to render some image into one of its buffers, then read
      this image out into a buffer object via glReadPixels.  Then, it can
      use this buffer object as a source of vertex data.


New Procedures and Functions

    None.


New Tokens

    Accepted by the <target> parameters of BindBuffer, BufferData,
    BufferSubData, MapBufferOES, MapBufferRangeEXT, UnmapBufferOES,
    FlushMappedBufferRangeEXT, GetBufferParameteriv, and
    GetBufferPointervOES:

        PIXEL_PACK_BUFFER_NV                        0x88EB
        PIXEL_UNPACK_BUFFER_NV                      0x88EC

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    GetFloatv:

        PIXEL_PACK_BUFFER_BINDING_NV                0x88ED
        PIXEL_UNPACK_BUFFER_BINDING_NV              0x88EF


Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

 -- Section 2.9 "Buffer Objects"

    Replace the first two paragraphs with:

    "The vertex data arrays described in section 2.8 are stored in
    client memory.  It is sometimes desirable to store frequently accessed
    client data, such as vertex array and pixel data, in high-performance
    server memory.  GL buffer objects provide a mechanism for clients to
    use to allocate, initialize, and access such memory."

    The name space for buffer objects is the unsigned integer, with zero
    reserved for the GL.  A buffer object is created by binding an unused
    name to a buffer target.  The binding is effected by calling

       void BindBuffer(enum target, uint buffer);

    <target> must be one of ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER,
    PIXEL_UNPACK_BUFFER_NV, or PIXEL_PACK_BUFFER_NV.  The ARRAY_BUFFER
    target is discussed in section 2.9.1  The ELEMENT_ARRAY_BUFFER target
    is discussed in section 2.9.2.  The PIXEL_UNPACK_BUFFER_NV and
    PIXEL_PACK_BUFFER_NV targets are discussed later in sections 3.7.1 and
    4.3.  If the buffer object named <buffer> has not been
    previously bound or has been deleted since the last binding, the
    GL creates a new state vector, initialized with a zero-sized memory
    buffer and comprising the state values listed in table 2.5."

    Replace the 5th paragraph with:

    "Initially, each buffer object target is bound to zero.  There is
    no buffer object corresponding to the name zero so client attempts
    to modify or query buffer object state for a target bound to zero
    generate an INVALID_OPERATION error."

    Replace the phrase listing the valid targets for BufferData in the
    9th paragraph with:

    "with <target> set to one of ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER,
    PIXEL_UNPACK_BUFFER_NV, or PIXEL_PACK_BUFFER_NV,"

    In the 10th paragraph describing buffer object usage modes, replace
    the phrase "specified once" with "specified once per repetition of
    the usage pattern" for the STREAM_* and STATIC_* usage values.

    Also in the 10th paragraph describing buffer object usage modes,
    replace the phrases "of a GL drawing command." and "for GL drawing
    commands." with "for GL drawing and image specification commands." for
    the *_DRAW usage values.

    Replace the phrase listing the valid targets for BufferSubData with:

    "with <target> set to one of ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER,
    PIXEL_UNPACK_BUFFER_NV, or PIXEL_PACK_BUFFER_NV."

    Replace the phrase listing the valid targets for MapBufferOES with:

    "with <target> set to one of ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER,
    PIXEL_UNPACK_BUFFER_NV, or PIXEL_PACK_BUFFER_NV."

    Replace the phrase listing the valid targets for UnmapBufferOES with:

    "with <target> set to one of ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER,
    PIXEL_UNPACK_BUFFER_NV, or PIXEL_PACK_BUFFER_NV."

    Replace the phrase listing the valid targets for MapBufferRangeEXT
    with:

    "with <target> set to one of ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER,
    PIXEL_UNPACK_BUFFER_NV, or PIXEL_PACK_BUFFER_NV."

    Replace the phrase listing the valid targets for
    FlushMappedBufferRangeEXT with:

    "with <target> set to one of ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER,
    PIXEL_UNPACK_BUFFER_NV, or PIXEL_PACK_BUFFER_NV."

 -- Section 2.9.2 "Array Indices in Buffer Objects"

    Delete the 3rd paragraph that explains how the ELEMENT_ARRAY_BUFFER
    target is acceptable for the commands specified in section 2.9.
    The updated section 2.9 language already says this.

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

 -- Section 3.6 "Pixel Rectangles"

    Replace the last two paragraphs with:

    "This section describes only how these rectangles are defined in
     buffer object and client memory, and the steps involved in
     transferring pixel rectangles from buffer object or client memory
     to the GL or vice-versa.
     Parameters controlling the encoding of pixels in buffer object or
     client memory (for reading and writing) are set with the command
     PixelStorei."

 -- Rename Section 3.6.1 "Pixel Storage Modes and Pixel Buffer Objects"

    Add to the end of the section:

    "In addition to storing pixel data in client memory, pixel data
    may also be stored in buffer objects (described in section 2.9).
    The current pixel unpack and pack buffer objects are designated
    by the PIXEL_UNPACK_BUFFER_NV and PIXEL_PACK_BUFFER_NV targets
    respectively.

    Initially, zero is bound for the PIXEL_UNPACK_BUFFER_NV, indicating
    that image specification commands such as TexImage*D source their
    pixels from client memory pointer parameters.  However, if a non-zero
    buffer object is bound as the current pixel unpack buffer, then
    the pointer parameter is treated as an offset into the designated
    buffer object."


 -- Section 3.6.2 "Transfer of Pixel Rectangles", page 61.

    Change the 1st sentence of the 1st paragraph to read:

    "The process of transferring pixels encoded in buffer object
     or client memory is diagrammed in figure 3.5."

    Change the 4th sentence of the 2nd paragraph to read:

    "<data> refers to the data to be transferred."

    [data is no longer necessarily a pointer.]

    Change the initial phrase in the 1st sentence of the 1st paragraph
    in subsection "Unpacking" to read:

    "Data are taken from the currently bound pixel unpack buffer or
    client memory as a sequence of..."

    Insert this paragraph after the 1st paragraph in subsection 
    "Unpacking":

    "If a pixel unpack buffer is bound (as indicated by a non-zero
    value of PIXEL_UNPACK_BUFFER_BINDING_NV), <data> is an offset
    into the pixel unpack buffer and the pixels are unpacked from the
    buffer relative to this offset; otherwise, <data> is a pointer to
    a block client memory and the pixels are unpacked from the client
    memory relative to the pointer.  If a pixel unpack buffer object
    is bound and unpacking the pixel data according to the process
    described below would access memory beyond the size of the pixel
    unpack buffer's memory size, INVALID_OPERATION results.  If a pixel
    unpack buffer object is bound and <data> is not evenly divisible
    into the number of basic machine units needed to store in memory the
    corresponding GL data type from table 3.4 for the <type> parameter,
    INVALID_OPERATION results."

 -- Section 3.7.1 "Texture Image Specification", page 66.

    Replace the last phrase in the 2nd to last sentence in the 1st
    paragraph with:

    "and a reference to the image data in the currently bound pixel unpack
    buffer or client memory, as described in section 3.6.2."

    Replace the 1st sentence in the 9th paragraph with:

    "The image itself (referred to by <data>) is a sequence of groups
    of values."

    Replace the last paragraph with:

    "If the data argument of TexImage2D is a NULL pointer, and the
     pixel unpack buffer object is zero, a two- or three-dimensional
     texel array is created with the specified target, level, internalformat,
     border, width, height, and depth, but with unspecified image contents.
     In this case no pixel values are accessed in client memory, and no pixel
     processing is performed. Errors are generated, however, exactly as though
     the data pointer were valid. Otherwise if the pixel unpack buffer object
     is non-zero, the data argument is treatedly normally to refer to the
     beginning of the pixel unpack buffer object's data."

 -- Section 3.7.3 "Compressed Texture Images", page 73.

    Replace the 3rd sentence of the 2nd paragraph with:

    "<data> refers to compressed image data stored in the compressed
    image format corresponding to internalformat.  If a pixel
    unpack buffer is bound (as indicated by a non-zero value of
    PIXEL_UNPACK_BUFFER_BINDING_NV), <data> is an offset into the
    pixel unpack buffer and the compressed data is read from the buffer
    relative to this offset; otherwise, <data> is a pointer to a block
    client memory and the compressed data is read from the client memory
    relative to the pointer."
    
    Replace the 2nd sentence in the 3rd paragraph with:

    "Compressed texture images are treated as an array of <imageSize>
    ubytes relative to <data>.  If a pixel unpack buffer object is bound
    and data+imageSize is greater than the size of the pixel buffer,
    INVALID_OPERATION results."

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Frame Buffer)

 -- Section 4.3 "Reading Pixels", page 104.

    Replace the first paragraph with:
    
    "Pixels may be read from the framebuffer to pixel pack buffer or 
     client memory using the ReadPixels commands, as described below.
     Pixels may also be copied from pixel unpack buffer, client memory or
     the framebuffer to texture images in the GL using the TexImage2D and
     CopyTexImage2D commands, as described in section 3.7.1.

 -- Section 4.3.1 "Reading Pixels", page 104.

    Replace 1st sentence of the 1st paragraph with:

    "The method for reading pixels from the framebuffer and placing them in
    pixel pack buffer or client memory is diagrammed in figure 4.2."

    Add this paragraph after the 1st paragraph:

    "Initially, zero is bound for the PIXEL_PACK_BUFFER_NV, indicating
    that image read and query commands such as ReadPixels return
    pixels results into client memory pointer parameters.  However, if
    a non-zero buffer object is bound as the current pixel pack buffer,
    then the pointer parameter is treated as an offset into the designated
    buffer object."

    Rename "Placement in Client Memory" to "Placement in Pixel Pack
    Buffer or Client Memory".

    Insert this paragraph at the start of the newly renamed
    subsection "Placement in Pixel Pack Buffer or Client Memory":

    "If a pixel pack buffer is bound (as indicated by a non-zero value
    of PIXEL_PACK_BUFFER_BINDING_NV), <data> is an offset into the
    pixel pack buffer and the pixels are packed into the
    buffer relative to this offset; otherwise, <data> is a pointer to a
    block client memory and the pixels are packed into the client memory
    relative to the pointer.  If a pixel pack buffer object is bound and
    packing the pixel data according to the pixel pack storage state
    would access memory beyond the size of the pixel pack buffer's
    memory size, INVALID_OPERATION results.  If a pixel pack buffer object
    is bound and <data> is not evenly divisible into the number of basic
    machine units needed to store in memory the corresponding GL data type
    from table 3.5 for the <type> parameter, INVALID_OPERATION results."


Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    None


Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

--- Section 6.1.3 Enumerated Queries

    Change the 1st sentence of the 3rd paragraph to read:
    "The command
        void GetBufferParameteriv( enum target, enum value, T data );
    returns information about <target>, which may be one of ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER, PIXEL_PACK_BUFFER_NV, PIXEL_UNPACK_BUFFER_NV
    indicating the currently bound vertex array, element array, pixel pack
    and pixel unpack buffer object."

 -- Section 6.1.13 "Buffer Object Queries".

    (description of glGetBufferPointervOES)  
    Change the 2nd sentence of the 2nd paragraph to read:
    "<target> is ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, PIXEL_PACK_BUFFER_NV,
    or PIXEL_UNPACK_BUFFER_NV."


Errors

    INVALID_ENUM is generated if the <target> parameter of
    BindBuffer, BufferData, BufferSubData, MapBufferOES, UnmapBufferOES,
    GetBufferParameteriv or GetBufferPointervOES is not
    one of ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, PIXEL_PACK_BUFFER_NV,
    or PIXEL_UNPACK_BUFFER_NV.

    INVALID_OPERATION is generated if CompressedTexImage2D, 
    CompressedTexSubImage2D, TexImage2D or TexSubImage2D would unpack
    (read) data from the currently bound PIXEL_UNPACK_BUFFER_NV buffer
    object such that the memory reads required for the command would exceed
    the memory (data store) size of the buffer object.

    INVALID_OPERATION is generated if ReadPixels or ReadnPixelsEXT 
    would pack (write) data to the currently bound PIXEL_PACK_BUFFER_NV
    buffer object such that the memory writes required for the command would
    exceed the memory (data store) size of the buffer object.

    INVALID_OPERATION is generated by ReadPixels or ReadnPixelsEXT
    if the current PIXEL_PACK_BUFFER_BINDING_NV value is non-zero and the
    table/image/values/span/img/data parameter is not evenly divisible
    into the number of basic machine units needed to store in memory a
    datum indicated by the type parameter.

    INVALID_OPERATION is generated by TexImage2D or TexSubImage2D
    if current PIXEL_UNPACK_BUFFER_BINDING_NV value is non-zero and the data
    parameter is not evenly divisible into the number of basic machine
    units needed to store in memory a datum indicated by the type
    parameter.

Dependencies on OES_mapbuffer
    
    If OES_mapbuffer is not present, references to MapBufferOES and
    UnmapBufferOES should be ignored and language referring to mapped
    buffer objects should be removed.

Dependencies on EXT_map_buffer_range
    
    If EXT_map_buffer_range is not present, references to
    MapBufferRangeEXT anf FlushMappedBufferRangeEXT should be ignored.
    
New State

(table 6.13, Pixels, p. 147)

                                                         Initial
    Get Value                        Type   Get Command  Value    Sec    
    -------------------------------  ----   -----------  -------  ------ 
    PIXEL_PACK_BUFFER_BINDING_NV    Z+     GetIntegerv  0        4.3.1  
    PIXEL_UNPACK_BUFFER_BINDING_NV  Z+     GetIntegerv  0        3.6.1  



Usage Examples

    Convenient macro definition for specifying buffer offsets:

        #define BUFFER_OFFSET(i) ((char *)NULL + (i))

    Example 1: Render to vertex array:

        const int numberVertices = 100;

        // Create a buffer object for a number of vertices consisting of
        // 4 float values per vertex
        glGenBuffers(1, vertexBuffer);
        glBindBuffer(GL_PIXEL_PACK_BUFFER_NV, vertexBuffer);
        glBufferData(GL_PIXEL_PACK_BUFFER_NV, numberVertices*4,
                     NULL, GL_DYNAMIC_DRAW);

        // Render vertex data into 100x1 strip of framebuffer using a
        // shader program
        glUseProgram(program);
        renderVertexData();
        glBindProgramARB(FRAGMENT_PROGRAM_ARB, 0);

        // Read the vertex data back from framebuffer
        glReadPixels(0, 0, numberVertices, 1, GL_BGRA, GL_UNSIGNED_BYTE,
                     BUFFER_OFFSET(0));

        // Change the binding point of the buffer object to
        // the vertex array binding point
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, BUFFER_OFFSET(0));
        glDrawArrays(TRIANGLE_STRIP, 0, numberVertices);


    Example 2: Streaming textures

    Streaming textures using pixel buffer objects:

        const int texWidth = 256;
        const int texHeight = 256;
        const int texsize = texWidth * texHeight * 4;
        void *pboMemory, *texData;

        // Define texture level zero (without an image); notice the
        // explicit bind to the zero pixel unpack buffer object so that
        // pass NULL for the image data leaves the texture image
        // unspecified.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_NV, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texWidth, texHeight, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        // Create and bind texture image buffer object
        glGenBuffers(1, &texBuffer);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_NV, texBuffer);

        // Setup texture environment
        ...

        texData = getNextImage();

        while (texData) {

            // Reset the contents of the texSize-sized buffer object
            glBufferData(GL_PIXEL_UNPACK_BUFFER_NV, texSize, NULL,
                         GL_STREAM_DRAW);

            // Map the texture image buffer (the contents of which
            // are undefined due to the previous glBufferData)
            pboMemory = glMapBufferOES(GL_PIXEL_UNPACK_BUFFER_NV,
                                    GL_WRITE_ONLY);

            // Modify (sub-)buffer data
            memcpy(pboMemory, texData, texsize);

            // Unmap the texture image buffer
            glUnmapBufferOES(GL_PIXEL_UNPACK_BUFFER_NV);

            // Update (sub-)teximage from texture image buffer
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texWidth, texHeight,
                            GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_OFFSET(0));

            // Draw textured geometry
            ...

            texData = getNextImage();
        }

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_NV, 0);


    Example 3: Asynchronous glReadPixels

    Traditional glReadPixels:

        const int imagewidth = 640;
        const int imageheight = 480;
        GLubyte readBuffer[imagewidth*imageheight*4];

        // Render to framebuffer
        renderScene()

        // Read image from framebuffer
        glReadPixels(0, 0, imagewidth, imageheight, GL_RGBA,
                     GL_UNSIGNED_BYTE, readBuffer);

        // Process image when glReadPixels returns after reading the
        // whole buffer
        processImage(readBuffer);


    Asynchronous glReadPixels:

        const int imagewidth = 640;
        const int imageheight = 480;
        const int imageSize = imagewidth*imageheight*4;

        glGenBuffers(2, imageBuffers);

        glBindBuffer(GL_PIXEL_PACK_BUFFER_NV, imageBuffers[0]);
        glBufferData(GL_PIXEL_PACK_BUFFER_NV, imageSize / 2, NULL,
                     GL_STREAM_DRAW);

        glBindBuffer(GL_PIXEL_PACK_BUFFER_NV, imageBuffers[1]);
        glBufferData(GL_PIXEL_PACK_BUFFER_NV, imageSize / 2, NULL,
                     GL_STREAM_DRAW);

        // Render to framebuffer
        glDrawBuffer(GL_BACK);
        renderScene();

        // Bind two different buffer objects and start the glReadPixels
        // asynchronously. Each call will return directly after
        // starting the DMA transfer.
        glBindBuffer(GL_PIXEL_PACK_BUFFER_NV, imageBuffers[0]);
        glReadPixels(0, 0, imagewidth, imageheight/2, GL_RGBA,
                     GL_UNSIGNED_BYTE, BUFFER_OFFSET(0));

        glBindBuffer(GL_PIXEL_PACK_BUFFER_NV, imageBuffers[1]);
        glReadPixels(0, imageheight/2, imagewidth, imageheight/2, GL_RGBA,
                     GL_UNSIGNED_BYTE, BUFFER_OFFSET(0));

        // Process partial images.  Mapping the buffer waits for
        // outstanding DMA transfers into the buffer to finish.
        glBindBuffer(GL_PIXEL_PACK_BUFFER_NV, imageBuffers[0]);
        pboMemory1 = glMapBufferRangeEXT(GL_PIXEL_PACK_BUFFER_NV, 0,
                                         imageSize/2, GL_MAP_READ_BIT_EXT);
        processImage(pboMemory1);
        glBindBuffer(GL_PIXEL_PACK_BUFFER_NV, imageBuffers[1]);
        pboMemory2 = glMapBufferRangeEXT(GL_PIXEL_PACK_BUFFER_NV, 0,
                                         imageSize/2, GL_MAP_READ_BIT_EXT);
        processImage(pboMemory2);

        // Unmap the image buffers
        glBindBuffer(GL_PIXEL_PACK_BUFFER_NV, imageBuffers[0]);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER_NV);
        glBindBuffer(GL_PIXEL_PACK_BUFFER_NV, imageBuffers[1]);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER_NV);


Issues


    1)  Can a given buffer be used for both vertex and pixel data?

        RESOLVED: YES.  All buffers can be used with all buffer bindings,
        in whatever combinations the application finds useful.  Consider
        yourself warned, however, by the following issue.

    2)  May implementations make use of the target as a hint to select
        an appropriate memory space for the buffer?

        RESOLVED: YES, as long as such behavior is transparent to the
        application. Some implementations may choose different memory 
        spaces for different targets.
        In fact, one can imagine arbitrarily complicated heuristics for
        selecting the memory space, based on factors such as the target,
        the "usage" argument, and the application's observed behavior.

        While it is entirely legal to create a buffer object by binding
        it to GL_ARRAY_BUFFER and loading it with data, then using it
        with the GL_PIXEL_UNPACK_BUFFER_NV or GL_PIXEL_PACK_BUFFER_NV
        binding, such behavior is liable to confuse the driver and may
        hurt performance.  If the driver implemented the hypothetical
        heuristic described earlier, such a buffer might have already
        been located in AGP memory, and so the driver would have to choose
        between two bad options: relocate the buffer into video memory, or
        accept lower performance caused by streaming pixel data from AGP.

    3)  Should the INVALID_OPERATION error be generated if a pixel
        command would access data outside the range of the bound PBO?

        RESOLVED:  YES.  This requires considering the command parameters
        (such as width/height/depth/format/type/pointer), the current
        pixel store (pack/unpack) state, and the command operation itself
        to determine the maximum addressed byte for the pixel command.

        This behavior should increase the reliability of using PBO and
        guard against programmer mistakes.

        This is particularly important for glReadPixels where returning
        data into a region outside the PBO could cause corruption of
        application memory.

        Such bounds checking is substantially more expensive for VBO
        accesses because bounds checking on a per-vertex element basis
        for each of multiple enabled vertex arrays prior to performing
        the command compromises the performance justification of VBO.

     4) If a pixel command with a bound PBO accesses data outside the
        range of the PBO, thereby generating a GL_INVALID_OPERATION error,
        can the pixel command end up being partially processed?

        RESOLVED:  NO.  As for all GL errors excepting GL_OUT_OF_MEMORY
        situations, "the command generating the error is ignored so that
        it has no effect on GL state or framebuffer contents."

        This means implementations must determine before the pixel command
        is performed whether the resulting read or write operations on
        the bound PBO will exceed the size of the PBO.

        This means an implementation is NOT allowed to detect out of
        bounds accesses in the middle of performing the command.
  
     5) Should an INVALID_OPERATION error be generated if the offset
        within a pixel buffer to a datum comprising of N basic machine
        units is not a multiple of N?

        RESOLVED:  YES.  This was stated for VBOs but no error was
        defined if the rule was violated.  Perhaps this needs to be
        better specified for VBO.

        For PBO, it is reasonable and cheap to enforce the alignment rule.
        For pixel commands it means making sure the offset is evenly
        divisible by the component or group size in basic machine units.

        This check is independent of the pixel store state because the
        pixel store state is specified in terms of pixels (not basic
        machine units) so pixel store addressing cannot create an
        unaligned access as long as the base offset is aligned.

        Certain commands (specifically,
        glCompressedTexImage2D, glCompressedTexSubImage2D) are not
        affected by this error because the data accessed is addressed
        at the granularity of basic machine units.

     6) Various commands do not make explicit reference to supporting
        packing or unpacking from a pixel buffer object but rather specify
        that parameters are handled in the same manner as glReadPixels,
        or the glCompressedTexImage commands.  So do such
        commands (example: glCompressedTexSubImage2D) use pixel buffers?

        RESOLVED:  YES.  Commands that have their behavior defined based
        on commands that read or write from pixel buffers will themselves
        read or write from pixel buffers.  Relying on this reduces the
        amount of specification language to be updated.

     7) What is the complete list of commands that can unpack (read)
        pixels from the current pixel unpack buffer object?

            glCompressedTexImage2D
            glCompressedTexSubImage2D
            glTexImage2D
            glTexSubImage2D

     8) What is the complete list of commands that can pack (write)
        pixels into the current pixel pack buffer object?

            glReadPixels

     9) Prior to this extension, passing zero for the data argument of
        glTexImage2D defined a texture image level without supplying an image.
        How does this behavior change with this extension?

        RESOLVED:  The "unspecified image" behavior of the glTexImage
        calls only applies when bound to a zero pixel unpack buffer
        object.

        When bound to a non-zero pixel unpack buffer object, the data
        argument to these calls is treated as an offset rather than
        a pointer so zero is a reasonable and even likely value that
        corresponds to the very beginning of the buffer object's data.

        So to create a texture image level with unspecified image data,
        you MUST bind to the zero pixel unpack buffer object.

        See the ammended language at the end of section 3.7.1.

    10) How does this extension support video frame grabbers?

        RESOLVED:  This extension extends buffer objects so they can
        operate with pixel commands, rather than just vertex array
        commands.

        We anticipate that a future extension may provide a mechanism
        for transferring video frames from video frame grabber hardware
        or vertices from motion capture hardware (or any other source
        of aquired real-time data) directly into a buffer object to
        eliminate a copy.  Ideally, such transfers would be possible
        without requiring mapping of the buffer object.  But this
        extension does not provide such functionality.

        We anticipate such functionality to involve binding a buffer
        object to a new target type, configuring a source (or sink) for
        data (video frames, motion capture vertex sets, etc.), and then
        commands to initiate data transfers to the bound buffer object.
        
    11) Is this the "right" way to expose render-to-vertex-array?

        DISCUSSION:  You can use this extension to render an image
        into a framebuffer, copy the pixels into a buffer object with
        glReadPixels, and then configure vertex arrays to source the pixel
        data as vertex attributes.  This necessarily involves a copy
        from the framebuffer to the buffer object.  Future extensions
        may provide mechanisms for copy-free render-to-vertex-array
        capabilities but that is not a design goal of this extension.
        
Revision History

    3   04/27/2020 fix example code: GL_STREAM_READ is not available in ES2.0
                   glMapBufferOES does not allow reading from the mapped pointer.
    2   10/23/2012 more cleanup, interaction with EXT_map_buffer_range
    1   04/19/2012 initial revision
        - took ARB_pixel_buffer_object, stripped everything not applicable to 
          ES, changed references to tables and sections; changed all wording
          to fit ES' language.

