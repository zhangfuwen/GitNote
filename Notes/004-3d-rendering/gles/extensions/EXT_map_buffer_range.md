# EXT_map_buffer_range

Name

    EXT_map_buffer_range

Name Strings

    GL_EXT_map_buffer_range

Contributors

    Contributors to ARB_map_buffer_range desktop OpenGL extension from which 
    this extension borrows heavily

Contact

    Benj Lipchak (lipchak 'at' apple 'dot' com)

Status

    Complete

Version

    Last Modified Date: August 21, 2014
    Author Revision: 4

Number

    OpenGL ES Extension #121

Dependencies

    OpenGL ES 1.1 or OpenGL ES 2.0 is required.
    
    OES_mapbuffer is required.

    This specification is written against the OpenGL ES 2.0.25 specification.

Overview

    EXT_map_buffer_range expands the buffer object API to allow greater
    performance when a client application only needs to write to a sub-range
    of a buffer object. To that end, this extension introduces two new buffer
    object features: non-serialized buffer modification and explicit sub-range
    flushing for mapped buffer objects.

    OpenGL requires that commands occur in a FIFO manner meaning that any
    changes to buffer objects either block until the data has been processed by
    the OpenGL pipeline or else create extra copies to avoid such a block.  By
    providing a method to asynchronously modify buffer object data, an
    application is then able to manage the synchronization points themselves
    and modify ranges of data contained by a buffer object even though OpenGL
    might still be using other parts of it.

    This extension also provides a method for explicitly flushing ranges of a
    mapped buffer object so OpenGL does not have to assume that the entire
    range may have been modified.  Further, it allows the application to more
    precisely specify its intent with respect to reading, writing, and whether
    the previous contents of a mapped range of interest need be preserved
    prior to modification.

New Procedures and Functions

    void *MapBufferRangeEXT(enum target, intptr offset, sizeiptr length,
        bitfield access);

    void FlushMappedBufferRangeEXT(enum target, intptr offset, 
        sizeiptr length);


New Tokens

    Accepted by the <access> parameter of MapBufferRangeEXT:

    MAP_READ_BIT_EXT                  0x0001
    MAP_WRITE_BIT_EXT                 0x0002
    MAP_INVALIDATE_RANGE_BIT_EXT      0x0004
    MAP_INVALIDATE_BUFFER_BIT_EXT     0x0008
    MAP_FLUSH_EXPLICIT_BIT_EXT        0x0010
    MAP_UNSYNCHRONIZED_BIT_EXT        0x0020


Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL ES Operation)

    Add to the end of Section 2.9 "Buffer Objects" (p. 24):

    All or part of the data store of a buffer object may be mapped into the
    client's address space by calling

    void *MapBufferRangeEXT(enum target, intptr offset, sizeiptr length,
        bitfield access);

    with <target> set to ARRAY_BUFFER or ELEMENT_ARRAY_BUFFER. 
    <offset> and <length> indicate the range of data in the
    buffer object that is to be mapped, in terms of basic machine units.
    <access> is a bitfield containing flags which describe the requested
    mapping. These flags are described below.

    If no error occurs, a pointer to the beginning of the mapped range is
    returned once all pending operations on that buffer have completed, and 
    may be used to modify and/or query the corresponding range of the buffer, 
    according to the following flag bits set in <access>:

    * MAP_READ_BIT_EXT indicates that the returned pointer may be used to 
    read buffer object data. No GL error is generated if the pointer is used 
    to query a mapping which excludes this flag, but the result is undefined 
    and system errors (possibly including program termination) may occur.

    * MAP_WRITE_BIT_EXT indicates that the returned pointer may be used to 
    modify buffer object data. No GL error is generated if the pointer is used
    to modify a mapping which excludes this flag, but the result is undefined 
    and system errors (possibly including program termination) may occur.

    Pointer values returned by MapBufferRangeEXT may not be passed as 
    parameter values to GL commands. For example, they may not be used to 
    specify array pointers, or to specify or query pixel or texture image 
    data; such actions produce undefined results, although implementations 
    may not check for such behavior for performance reasons.
    
    Mappings to the data stores of buffer objects may have nonstandard 
    performance characteristics. For example, such mappings may be marked as 
    uncacheable regions of memory, and in such cases reading from them may be 
    very slow. To ensure optimal performance, the client should use the 
    mapping in a fashion consistent with the values of BUFFER_USAGE and 
    <access>. Using a mapping in a fashion inconsistent with these values is 
    liable to be multiple orders of magnitude slower than using normal memory.

    The following optional flag bits in <access> may be used to modify the 
    mapping:

    * MAP_INVALIDATE_RANGE_BIT_EXT indicates that the previous contents of 
    the specified range may be discarded. Data within this range are undefined
    with the exception of subsequently written data. No GL error is generated 
    if subsequent GL operations access unwritten data, but the result is 
    undefined and system errors (possibly including program termination) may 
    occur. This flag may not be used in combination with MAP_READ_BIT_EXT.

    * MAP_INVALIDATE_BUFFER_BIT_EXT indicates that the previous contents of 
    the entire buffer may be discarded. Data within the entire buffer are 
    undefined with the exception of subsequently written data. No GL error is 
    generated if subsequent GL operations access unwritten data, but the 
    result is undefined and system errors (possibly including program 
    termination) may occur. This flag may not be used in combination with 
    MAP_READ_BIT_EXT.

    * MAP_FLUSH_EXPLICIT_BIT_EXT indicates that one or more discrete 
    subranges of the mapping may be modified. When this flag is set, 
    modifications to each subrange must be explicitly flushed by calling 
    FlushMappedBufferRangeEXT. No GL error is set if a subrange of the 
    mapping is modified and not flushed, but data within the corresponding 
    subrange of the buffer is undefined. This flag may only be used in 
    conjunction with MAP_WRITE_BIT_EXT. When this option is selected, 
    flushing is strictly limited to regions that are explicitly indicated 
    with calls to FlushMappedBufferRangeEXT prior to unmap; if this
    option is not selected UnmapBufferOES will automatically flush the entire
    mapped range when called.

    * MAP_UNSYNCHRONIZED_BIT_EXT indicates that the GL should not attempt 
    to synchronize pending operations on the buffer prior to returning from
    MapBufferRangeEXT. No GL error is generated if pending operations which 
    source or modify the buffer overlap the mapped region, but the result of 
    such previous and any subsequent operations is undefined.

    A successful MapBufferRangeEXT sets buffer object state values as shown 
    in table 2.mbr.
    
    Name                 Value
    -------------------  -----
    BUFFER_ACCESS_FLAGS  <access>
    BUFFER_MAPPED        TRUE
    BUFFER_MAP_POINTER   pointer to the data store
    BUFFER_MAP_OFFSET    <offset>
    BUFFER_MAP_LENGTH    <length>
    
    Table 2.mbr: Buffer object state set by MapBufferRangeEXT.

    If an error occurs, MapBufferRangeEXT returns a NULL pointer.
    
    An INVALID_VALUE error is generated if <offset> or <length> is negative, 
    if <offset> + <length> is greater than the value of BUFFER_SIZE, or if 
    <access> has any bits set other than those defined above.
    
    An INVALID_OPERATION error is generated for any of the following 
    conditions:
    
    * <length> is zero.
    * The buffer is already in a mapped state.
    * Neither MAP_READ_BIT_EXT nor MAP_WRITE_BIT_EXT is set.
    * MAP_READ_BIT_EXT is set and any of MAP_INVALIDATE_RANGE_BIT_EXT, 
      MAP_INVALIDATE_BUFFER_BIT_EXT, or MAP_UNSYNCHRONIZED_BIT_EXT is set.
    * MAP_FLUSH_EXPLICIT_BIT_EXT is set and MAP_WRITE_BIT_EXT is not set.
    
    An OUT_OF_MEMORY error is generated if MapBufferRangeEXT fails because
    memory for the mapping could not be obtained.

    No error is generated if memory outside the mapped range is modified or
    queried, but the result is undefined and system errors (possibly including
    program termination) may occur.

    If a buffer is mapped with the MAP_FLUSH_EXPLICIT_BIT_EXT flag, 
    modifications to the mapped range may be indicated by calling

    void FlushMappedBufferRangeEXT(enum target, intptr offset, 
        sizeiptr length);

    with <target> set to ARRAY_BUFFER or ELEMENT_ARRAY_BUFFER. <offset> and 
    <length> indicate a modified subrange of the mapping, in basic machine 
    units. The specified subrange to flush is relative to the start of the 
    currently mapped range of buffer. FlushMappedBufferRangeEXT may be 
    called multiple times to indicate distinct subranges of the mapping which 
    require flushing.

    An INVALID_VALUE error is generated if <offset> or <length> is negative, 
    or if <offset> + <length> exceeds the size of the mapping.

    An INVALID_OPERATION error is generated if zero is bound to <target>.

    An INVALID_OPERATION error is generated if the buffer bound to <target> 
    is not mapped, or is mapped without the MAP_FLUSH_EXPLICIT_BIT_EXT flag.

New Implementation Dependent State

    None

Usage Examples

    /* bind and initialize a buffer object */
    int size = 65536;
    glBindBuffer(GL_ARRAY_BUFFER, 1);
    glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);

/* the following are not meant to be executed as a group, since there are no
 * unmap calls shown here - they are meant to show different combinations of
 * map options in conjunction with MapBufferRangeEXT and 
 * FlushMappedBufferRangeEXT.
 */

    /* Map the entire buffer with read and write
     * (identical semantics to MapBufferOES).
     */
    void *ptr = glMapBufferRangeEXT(GL_ARRAY_BUFFER, 0, size, 
        MAP_READ_BIT_EXT | MAP_WRITE_BIT_EXT);

    /* Map the entire buffer as write only.
     */
    void *ptr = glMapBufferRangeEXT(GL_ARRAY_BUFFER, 0, size, 
        MAP_WRITE_BIT_EXT);


    /* Map the last 1K bytes of the buffer as write only.
     */
    void *ptr = glMapBufferRangeEXT(GL_ARRAY_BUFFER, size-1024, 1024, 
        MAP_WRITE_BIT_EXT);


    /* Map the last 1K bytes of the buffer as write only, and invalidate the 
     * range. Locations within that range can assume undefined values.
     * Locations written while mapped take on new values as expected.
     * No changes occur outside the range mapped.
     */
    void *ptr = glMapBufferRangeEXT(GL_ARRAY_BUFFER, size-1024, 1024, 
        MAP_WRITE_BIT_EXT | MAP_INVALIDATE_RANGE_BIT_EXT);


    /* Map the first 1K bytes of the buffer as write only, and invalidate the 
     * entire buffer. All locations within the buffer can assume undefined 
     * values. Locations written while mapped take on new values as expected.
     */
    void *ptr = glMapBufferRangeEXT(GL_ARRAY_BUFFER, 0, 1024, 
        MAP_WRITE_BIT_EXT | MAP_INVALIDATE_BUFFER_BIT_EXT);


    /* Map the first 32K bytes of the buffer as write only, and invalidate 
     * that range. Indicate that we will explicitly inform GL which ranges are 
     * actually written. Locations within that range can assume undefined 
     * values. Only the locations which are written and subsequently flushed 
     * are guaranteed to take on defined values.
     * Write data to the first 8KB of the range, then flush it.
     * Write data to the last 8KB of the range, then flush it.
     */
    void *ptr = glMapBufferRangeEXT(GL_ARRAY_BUFFER, 0, 32768, 
        MAP_WRITE_BIT_EXT | MAP_INVALIDATE_RANGE_BIT_EXT | 
        MAP_FLUSH_EXPLICIT_BIT_EXT);

    memset(ptr, 0x00, 8192);           /* write zeroes to first 8KB of range */
    glFlushMappedBufferRangeEXT(GL_ARRAY_BUFFER, 0, 8192);

    memset(((char*)ptr)+24576, 0xFF, 8192);/* write FFs to last 8KB of range */
    glFlushMappedBufferRangeEXT(GL_ARRAY_BUFFER, 24576, 8192);


    /* Map the entire buffer for write - unsynchronized.
     * GL will not block for prior operations to complete.  Application must
     * use other synchronization techniques to ensure correct operation.
     */
    void *ptr = glMapBufferRangeEXT(GL_ARRAY_BUFFER, 0, size, 
        MAP_WRITE_BIT_EXT | MAP_UNSYNCHRONIZED_BIT_EXT);


Revision History

    Version 4, 2014/08/21 - Fix typo OES_map_buffer -> OES_mapbuffer.
    Version 3, 2012/06/21 - Recast from APPLE to multivendor EXT
    Version 2, 2012/06/18 - Correct spec to indicate ES 1.1 may also be okay.
    Version 1, 2012/06/01 - Conversion from ARB_map_buffer_range to 
                            APPLE_map_buffer_range for ES.
