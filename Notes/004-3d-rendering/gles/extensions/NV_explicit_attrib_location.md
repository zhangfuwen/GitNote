# NV_explicit_attrib_location

Name

    NV_explicit_attrib_location

Name Strings

    GL_NV_explicit_attrib_location

Contributors

    Contributors to ARB_explicit_attrib_location
    Mathias Heyer, NVIDIA

Contact

    Greg Roth, NVIDIA (groth 'at' nvidia.com)

Status

    Shipping on Tegra

Version

    Last Modified Date:         September 20, 2013
    Revision:                   2

Number

    OpenGL ES Extension #159

Dependencies

    Requires OpenGL ES 2.0.

    Written based on the wording of the OpenGL ES 2.0.25 Full Specification
    (November 2, 2010).

    Written based on the wording of The OpenGL ES Shading Language 1.0.17
    Specification (May 12, 2009).    

Overview

    This extension provides a method to pre-assign attribute locations
    to named vertex shader inputs.  This allows applications to globally
    assign a particular semantic meaning, such as diffuse color or
    vertex normal, to a particular attribute location without knowing
    how that attribute will be named in any particular shader.

New Procedures and Functions

    None

New Tokens

    None
    
Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    Section 2.10.4 "Shader Variables", subsection "Vertex Attributes"

    Modify the first paragraph to read:

    "Vertex shaders can define named attribute variables, which are
    bound to the generic vertex attributes that are set by
    VertexAttrib*. This binding can be specified by the application
    before the program is linked, either through BindAttribLocation
    (described below) or explicitly within the shader text, or
    automatically assigned by the GL when the program is linked."

    Modify the third paragraph describing BindAttribLocation to read:
 
    "When a program is linked, any active attributes without a binding
    specified either through BindAttribLocation or explicitly set within
    the shader text will automatically be bound to vertex attributes by
    the GL. Such bindings can be queried using the command
    GetAttribLocation. LinkProgram will fail if the assigned binding of
    an active attribute variable would cause the GL to reference a
    nonexistent generic attribute (one greater than or equal to the
    value of MAX_VERTEX_ATTRIBS). LinkProgram will fail if the attribute
    bindings specified either through BindAttribLocation or explicitly
    set within the shader text do not leave enough space to assign a
    location for an active matrix attribute, which requires multiple
    contiguous generic attributes. If an active attribute has a binding
    explicitly set within the shader text and a different binding
    assigned by BindAttribLocation, the assignment in the shader text is
    used."

Additions to OpenGL ES Shading Language 1.00 Specification

    Including the following line in a shader can be used to control
    the language feature described in this extension:

      #extension GL_NV_explicit_attrib_location : <behavior>

    where <behavior> is as described in section 3.4.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

      #define GL_NV_explicit_attrib_location 1

Section 4.3.3 "Attribute"

    Add new section 4.3.3.1 "Attribute Layout Qualifiers"

    "Vertex shaders allow location layout qualifiers on attribute
    variable declarations. They can appear with an individual variable
    declared with an attribute qualifier:

        <layout-qualifier> attribute <declaration>;

    Layouts qualified declarations can only be made at global scope,
    and only on attribute variable declarations.

    <layout-qualifier> expands to:

        layout-qualifier :
            layout (<layout-qualifier-id>)

        <layout-qualifier-id> :
            location = <integer-constant>

    Only one argument is accepted.  For example,

      layout(location = 3) attribute vec4 normal;

    will establish that the vertex shader attribute <normal> is copied
    in from vector location number 3.

    If the named vertex shader input has a scalar or vector type, it
    will consume a single location.

    If the named vertex shader attribute is a matrix, it will be
    assigned multiple locations starting with the location specified.
    The number of locations assigned for each matrix will be equal to
    the number of columns in the matrix  For example,

        layout(location = 9) attribute mat4 transform;

    will establish that input <transform> is assigned to vector location
    numbers 9-12.

    If an attribute variable with no location assigned in the shader
    text has a location specified through the OpenGL ES API, the API-
    assigned location will be used.  Otherwise, such variables will be
    assigned a location by the linker.  See section 2.10.4 of the OpenGL
    ES Specification for more details.

Errors

    None, see issue #1.

New State

    None.

New Implementation Dependent State

    None.

Issues

    1. How should the error be reported when the attribute location
       specified in the shader source is larger than MAX_VERTEX_ATTRIBUTES?
       
       RESOLVED.  Generate a link error.  The existing spec language already
       covers this case:
       
       "LinkProgram will fail if the assigned binding of an active attribute
       variable would cause the GL to reference a non-existent generic
       attribute (one greater than or equal to MAX_VERTEX_ATTRIBS)."

    2. What happens when the shader text binds an input to a
       particular attribute location and the same attribute location is
       bound to a different attribute via the API?

       RESOLVED.  The setting in the shader is always used.

    3. Should layout-qualifier-id be index or location?

       RESOLVED.  location.  The API uses both.  <index> is used as the
       parameter name to VertexAttribPointer and BindAttribLocation, but
       "location" is used in the name of BindAttribLocation and
       GetAttribLocation.  However, there is some expectation that <index> may
       be used for another purpose later.

    4. The GL spec allows BindAttribLocation to be called before attaching
       shaders or linking.  If an application does this and specifies a
       layout, which takes precedence?

       RESOLVED.  The setting the shader is always used.

           The three options that were considered:

           a. The setting from the API, if specified, always wins.

           b. The setting from the shader, if specified, always wins.

           c. The setting is order dependent.  If the shader is
              attached after the API setting is made, the shader
              layout is used.  If the API setting is made after the
              shader is attached, the API setting is used.

    5. What happens if an input or output variable is declared in two
       shader objects with conflicting attribute locations?

       RESOLVED.  Not relevant to ES.

    6. What happens if an input or output variable is declared in two
       shader objects with an attribute location assigned in one shader but
       not the other.

       RESOLVED. Not relevant to ES.


Revision History

    Rev.    Date        Author       Changes
    ----   ----------   ---------    ------------------------------------
     2     09/20/2013   dkoch        minor edits for publishing
     1     04/25/2012   groth        First revision based on
                                     ARB_explicit_attrib_location.
