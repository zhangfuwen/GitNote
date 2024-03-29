# EXT_instanced_arrays

Name

    EXT_instanced_arrays

Name Strings

    GL_EXT_instanced_arrays

Contributors

    Contributors to ARB_instanced_arrays desktop OpenGL extension 
        from which this extension borrows heavily
    Abhijit Bhelande, Apple
    Benj Lipchak, Apple

Contact

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Status

    Complete

Version

    Last Modified Date:     June 26, 2013
    Revision:               2

Number

    OpenGL ES Extension #156

Dependencies

    OpenGL ES 2.0 is required.

    This extension is written against the OpenGL ES 2.0 Specification.
    
    OES_element_index_uint affects the definition of this extension.

Overview

    A common use case in GL for some applications is to be able to
    draw the same object, or groups of similar objects that share
    vertex data, primitive count and type, multiple times.  This 
    extension provides a means of accelerating such use cases while 
    reducing the number of API calls, and keeping the amount of 
    duplicate data to a minimum.
    
    This extension introduces an array "divisor" for generic
    vertex array attributes, which when non-zero specifies that the
    attribute is "instanced."  An instanced attribute does not
    advance per-vertex as usual, but rather after every <divisor>
    conceptual draw calls.
    
    (Attributes which aren't instanced are repeated in their entirety
    for every conceptual draw call.)
    
    By specifying transform data in an instanced attribute or series
    of instanced attributes, vertex shaders can, in concert with the 
    instancing draw calls, draw multiple instances of an object with 
    one draw call.

IP Status

    No known IP claims.

New Tokens

    Accepted by the <pname> parameters of GetVertexAttribfv and 
    GetVertexAttribiv:

        VERTEX_ATTRIB_ARRAY_DIVISOR_EXT                 0x88FE

New Procedures and Functions

    void VertexAttribDivisorEXT(uint index, uint divisor);
    void DrawArraysInstancedEXT(enum mode, int first, sizei count,
            sizei instanceCount);
    void DrawElementsInstancedEXT(enum mode, sizei count, enum type,
            const void *indices, sizei instanceCount);

Additions to Chapter 2 of the OpenGL ES 2.0 Specification

    Modify section 2.8 (Vertex Arrays), p. 21

    (insert before section Transferring Array Elements, p. 21)

    "The command
        
        void VertexAttribDivisorEXT(uint index, uint divisor);

    modifies the rate at which generic vertex attributes advance, which is 
    useful when rendering multiple instances of primitives in a single draw call
    (see DrawArraysInstancedEXT and DrawElementsInstancedEXT below). If 
    <divisor> is zero, the attribute at slot <index> advances once per vertex. 
    If <divisor> is non-zero, the attribute advances once per <divisor> 
    instances of the primitives being rendered. An attribute is referred to as 
    instanced if its <divisor> value is non-zero.
    
    An INVALID_VALUE error is generated if <index> is greater than or equal to 
    the value of MAX_VERTEX_ATTRIBS."

    (replace all occurrences of "DrawArrays or DrawElements" with "DrawArrays, 
    DrawElements, or the other Draw* commands", for example the first sentence 
    of Transferring Array Elements, p. 21)
    
    "When an array element i is transferred to the GL by DrawArrays, 
    DrawElements, or the other Draw* commands described below, each generic 
    attribute is expanded to four components."
    
    (replace second through fourth paragraphs of Transferring Array Elements)
    
    "The command

        void DrawArraysOneInstance(enum mode, int first, sizei count, 
                int instance);

    does not exist in the GL, but is used to describe functionality in the rest 
    of this section. This command constructs a sequence of geometric primitives
    by successively transferring elements for <count> vertices. Elements <first> 
    through <first> + <count> − 1 of each enabled non-instanced array are 
    transferred to the GL. <mode> specifies what kind of primitives are 
    constructed, as defined in section 2.6.1.
    
    If an enabled vertex attribute array is instanced (it has a non-zero 
    <divisor> as specified by VertexAttribDivisorEXT), the element that is 
    transferred to the GL, for all vertices, is given by:
    
         floor(instance / divisor)

    If an array corresponding to a generic attribute is not enabled, then the 
    corresponding element is taken from the current generic attribute state (see
    section 2.7). Otherwise, if an array is enabled, the corresponding current 
    generic attribute value is unaffected by the execution of 
    DrawArraysOneInstance.

    Specifying <first> < 0 results in undefined behavior. Generating the error 
    INVALID_VALUE is recommended in this case.
    
    The command

        void DrawArrays(enum mode, int first, sizei count);

    is equivalent to the command sequence

        DrawArraysOneInstance(mode, first, count, 0); 

    The command
        
        void DrawArraysInstancedEXT(enum mode, int first, sizei count, 
                sizei instanceCount);

    behaves identically to DrawArrays except that <instanceCount> instances of 
    the range of elements are executed and the value of <instance> advances 
    for each iteration. Those attributes that have non-zero values for 
    <divisor>, as specified by VertexAttribDivisorEXT, advance once every 
    <divisor> instances. It has the same effect as:
    
        if (mode, count, or instanceCount is invalid) 
            generate appropriate error
        else {
            for (i = 0; i < instanceCount; i++) {
                DrawArraysOneInstance(mode, first, count, i); 
            }
        }
            
    The command
        
        void DrawElementsOneInstance(enum mode, sizei count, enum type, 
                const void *indices, int instance);
            
    does not exist in the GL, but is used to describe functionality in the rest 
    of this section. This command constructs a sequence of geometric primitives 
    by successively transferring elements for <count> vertices. The ith element 
    transferred by DrawElementsOneInstance will be taken from element 
    <indices>[i] of each enabled non-instanced array, where <indices> specifies 
    the location in memory of the first index of the element array being 
    specified. <type> must be one of UNSIGNED_BYTE, UNSIGNED_SHORT, or 
    UNSIGNED_INT indicating that the index values are of GL type ubyte, ushort,
    or uint respectively. <mode> specifies what kind of primitives are 
    constructed, as defined in section 2.6.1.
    
    If an enabled vertex attribute array is instanced (it has a non-zero 
    <divisor> as specified by VertexAttribDivisorEXT), the element that is 
    transferred to the GL, for all vertices, is given by:
    
    floor(instance / divisor)

    If an array corresponding to a generic attribute is not enabled, then the 
    corresponding element is taken from the current generic attribute state (see
    section 2.7). Otherwise, if an array is enabled, the corresponding current 
    generic attribute value is unaffected by the execution of 
    DrawElementsOneInstance.

    The command
    
        void DrawElements(enum mode, sizei count, enum type, 
                const void *indices);
    
    behaves identically to DrawElementsOneInstance with the <instance> 
    parameter set to zero; the effect of calling
    
        DrawElements(mode, count, type, indices); 
    
    is equivalent to the command sequence:
        
        if (mode, count or type is invalid) 
            generate appropriate error
        else
            DrawElementsOneInstance(mode, count, type, indices, 0);
            
    The command
            
        void DrawElementsInstancedEXT(enum mode, sizei count, enum type, 
                const void *indices, sizei instanceCount);
            
    behaves identically to DrawElements except that <instanceCount> instances of
    the set of elements are executed and the value of <instance> advances 
    between each set. Instanced attributes are advanced as they do during 
    execution of DrawArraysInstancedEXT. It has the same effect as:
            
        if (mode, count, instanceCount, or type is invalid) 
            generate appropriate error
        else {
            for (int i = 0; i < instanceCount; i++) {
                DrawElementsOneInstance(mode, count, type, indices, i); 
            }
        }
        
    (append to first sentence of last paragraph of Transferring Array Elements)
    
    "..., and n integers representing vertex attribute divisors."

    (append to last sentence of last paragraph of Transferring Array Elements)
    
    "..., the divisors are each zero."

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    In section 6.1.8, add to the list of pnames accepted by GetVertexAttrib*v: 
    VERTEX_ATTRIB_ARRAY_DIVISOR_EXT

Dependencies on OES_element_index_uint
    
    If OES_element_index_uint is not supported, remove references to
    UNSIGNED_INT as a valid <type> for DrawElements*.
    
Errors

    INVALID_VALUE is generated by VertexAttribDivisorEXT if <index>
    is greater than or equal to MAX_VERTEX_ATTRIBS.

    INVALID_ENUM is generated by DrawElementsInstancedEXT if <type> is
    not one of UNSIGNED_BYTE, UNSIGNED_SHORT or UNSIGNED_INT.

    INVALID_VALUE is generated by DrawArraysInstancedEXT if <first>,
    <count>, or <instanceCount> is less than zero.

    INVALID_VALUE is generated by DrawElementsInstancedEXT if <count> or
    <instanceCount> is less than zero.

    INVALID_ENUM is generated by DrawArraysInstancedEXT or
    DrawElementsInstancedEXT if <mode> is not one of the kinds of primitives
    accepted by DrawArrays and DrawElements.


New State

    Changes to table 6.2, p. 136 (Vertex Array Data)
                                                                    Initial
    Get Value                          Type      Get Command        Value    Description          Sec.
    ---------                          ----      -----------        -------  -----------          ----
    VERTEX_ATTRIB_ARRAY_DIVISOR_EXT    16* x Z+  GetVertexAttribiv  0        Vertex attrib array  2.8
                                                                             instance divisor
Issues

    None
    
Revision History

    #1 November 11 2012, Abhijit Bhelande and Benj Lipchak
        - initial conversion from ARB to APPLE for ES2
    #2 June 26 2013, Benj Lipchak
        - promotion from APPLE to EXT
