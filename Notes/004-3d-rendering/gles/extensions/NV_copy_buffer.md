# NV_copy_buffer

Name

    NV_copy_buffer

Name Strings

    GL_NV_copy_buffer

Contact

    Mathias Heyer, NVIDIA Corporation (mheyer 'at' nvidia.com)

Contributors

    From ARB_copy_buffer:
    Rob Barris, Blizzard Entertainment
    Bruce Merry, ARM
    Eric Werness, NVIDIA
    Greg Roth, NVIDIA
    Daniel Koch, NVIDIA

Status

    Shipping on Tegra

Version

    Last Modified Date: September 20, 2013
    Author Revision: 3

Number

    OpenGL ES Extension #158

Dependencies

    Written based on the wording of the OpenGL ES 2.0.25 (Nov, 2010)
    specification.
    
    OES_mapbuffer extension affects the definition of this extension.

Overview

    This extension provides a mechanism to do an accelerated copy from one
    buffer object to another. This may be useful to load buffer objects
    in a "loading thread" while minimizing cost and synchronization effort
    in the "rendering thread."

New Tokens

    Accepted by the target parameters of BindBuffer, BufferData,
    BufferSubData, MapBufferOES, UnmapBufferOES, 
    GetBufferPointervOES, GetBufferParameteriv and CopyBufferSubDataNV:

    COPY_READ_BUFFER_NV                    0x8F36
    COPY_WRITE_BUFFER_NV                   0x8F37

New Procedures and Functions

    void CopyBufferSubDataNV(enum readtarget, enum writetarget,
                             intptr readoffset, intptr writeoffset,
                             sizeiptr size);

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (Rasterization)

    Add a new subsection "Copying Between Buffers" to section 2.9:

    All or part of one buffer object's data store may be copied to the
    data store of another buffer object by calling

    void CopyBufferSubDataNV(enum readtarget, enum writetarget,
                             intptr readoffset, intptr writeoffset,
                             sizeiptr size);

    with readtarget and writetarget each set to one of the targets
    ARRAY_BUFFER, COPY_READ_BUFFER_NV, COPY_WRITE_BUFFER_NV or
    ELEMENT_ARRAY_BUFFER. While any of these targets may be used,
    the COPY_READ_BUFFER_NV and COPY_WRITE_BUFFER_NV targets are
    provided specifically for copies, so that they can be done without
    affecting other buffer binding targets that may be in use.
    writeoffset and size specify the range of data in the buffer object
    bound to writetarget that is to be replaced, in terms of basic machine
    units. readoffset and size specify the range of data in the buffer
    object bound to readtarget that is to be copied to the corresponding
    region of writetarget.
    
    A buffer object is bound to COPY_READ_BUFFER_NV or COPY_WRITE_BUFFER_NV
    by calling BindBuffer with target set to COPY_READ_BUFFER_NV or
    COPY_WRITE_BUFFER_NV and buffer set to the name of the buffer object.
    If no corresponding buffer object exists, one is initialized as
    defined in section 2.9.
    
    The commands BufferData, BufferSubData, MapBufferOES, UnmapBufferOES, 
    GetBufferPointervOES, GetBufferParameteriv (section 6.1.3)  may be used
    with target set to COPY_READ_BUFFER_NV or COPY_WRITE_BUFFER_NV. In such
    event, these commands operate in the same fashion as described in section
    2.9 and 6.3.1 but on the buffer currently bound to target
    COPY_READ_BUFFER_NV or COPY_WRITE_BUFFER_NV respectively.

    An INVALID_VALUE error is generated if any of readoffset,
    writeoffset, or size are negative, if readoffset+size exceeds the
    size of the buffer object bound to readtarget, or if
    writeoffset+size exceeds the size of the buffer object bound to
    writetarget.

    An INVALID_VALUE error is generated if the same buffer object is
    bound to both readtarget and writetarget, and the ranges
    [readoffset, readoffset+size) and [writeoffset, writeoffset+size)
    overlap.

    An INVALID_OPERATION error is generated if zero is bound to
    readtarget or writetarget.
    
    An INVALID_OPERATION error is generated if the buffer objects
    bound to either readtarget or writetarget are mapped.

Additions to the AGL/EGL/GLX/WGL Specifications

    None

Errors

    The error INVALID_VALUE is generated by CopyBufferSubDataNV if
    readoffset, writeoffset, or size are less than zero, or if
    readoffset+size is greater than the value of BUFFER_SIZE of
    readtarget/readBuffer, or if writeoffset+size is greater than the
    value of BUFFER_SIZE of writetarget/writeBuffer.
    
    The error INVALID_OPERATION is generated by CopyBufferSubDataNV if
    either readtarget/readBuffer or writetarget/writeBuffer are mapped.

    The error INVALID_VALUE is generated by CopyBufferSubDataNV if
    readtarget/readBuffer and writetarget/writeBuffer are the same
    buffer object, and the ranges [readoffset, readoffset+size) and
    [writeoffset, writeoffset+size) overlap.

New State

    (add to table 6.2, Vertex Array State)

                                            Initial
    Get Value              Type    Get Command Value   Description                 Sec.    
    ----------------       ----    ----------- ------- --------------------------- ------  
    COPY_READ_BUFFER_NV    Z+      GetIntegerv 0       Buffer object bound to the  2.9     
                                                       copy buffer "read" binding
                                                       point
    COPY_WRITE_BUFFER_NV   Z+      GetIntegerv 0       Buffer object bound to the  2.9     
                                                       copy buffer "write"
                                                       binding point

Issues

    1) How is this extension useful?
        
    This can be a desirable replacement to BufferSubData if there are
    large updates that will pollute the CPU cache. If generating the data
    can be offloaded to another thread, then the CPU cost of the update
    in the rendering thread can be very small.

    Finally, if an implementation supports concurrent data transfers in
    one context/thread while doing rendering in another context/thread,
    this extension may be used to move data from system memory to video
    memory in preparation for copying it into another buffer in the
    rendering thread.
       

Dependencies on OES_mapbuffer

    If OES_mapbuffer is not present, references to MapBufferOES and
    UnmapBufferOES should be ignored and language referring to mapped
    buffer objects should be removed.
    
Revision History

    Revision 3, 2013/08/20
    - minor edits for publishing
    Revision 2, 2012/04/19
    - Added explicit interaction with BufferData, BufferSubData, MapBufferOES,
      UnmapBufferOES, GetBufferPointervOES and GetBufferParameteriv
    Revision 1, 2012/04/18
     - Initial draft by starting with a copy of ARB_copy_buffer and stripping
       it to ES
