# EXT_shader_non_constant_global_initializers

Name

    EXT_shader_non_constant_global_initializers

Name Strings

    GL_EXT_shader_non_constant_global_initializers

Contact

    Daniel Koch, NVIDIA (dkoch 'at' nvidia.com)

Contributors

    Sahil Parmar, NVIDIA
    Weiwan Liu, NVIDIA
    John Kessenich, Google
    Jeff Leger, Qualcomm

Status

    Complete

Version

    Last Modified Date: July 4, 2016
    Revision: 2

Number

    OpenGL ES Extension #264

Dependencies

    OpenGL ES Shading Language 1.00 is required.

    This extension is written against the OpenGL ES 3.20
    Shading Language (August 6, 2015) specification, but
    can apply to earlier versions.

Overview

    This extension adds the ability to use non-constant initializers for
    global variables in the OpenGL ES Shading Language specifications.
    This functionality is already present in the OpenGL Shading language
    specification.

New Procedures and Functions

    None.

New Tokens

    None.

Additions to the OpenGL ES 3.2 Specification

    None

Additions to the OpenGL ES Shading Language 3.20 Specification

    Including the following line in a shader can be used to control the
    language features described in this extension:

      #extension GL_EXT_shader_non_constant_global_initializers : <behavior>

    where <behavior> is as specified in section 3.4.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

      #define GL_EXT_shader_non_constant_global_initializers 1

    Modifications to Section 4.3 (Storage Qualifiers):

    Replace the last paragraph:

    "In declarations of global variables with no storage qualifier or
    with a const qualifier, any initializer must be a constant expression.
    Declarations of global variables with other storage qualifiers may not
    contain initializers. Global variables without storage qualifiers that
    are not initialized in their declaration or by the application will not
    be initialized by OpenGL ES, but rather will enter main() with undefined
    values."

    with the following paragraph:

    "Initializers in global declarations may only be used in declarations
    of global variables with either no storage qualifier, or a "const"
    qualifier. All such initializers will have been executed before, or
    on entry to, main(). Global variables without storage qualifiers that
    are not initialized in their declaration or by the application will not
    be initialized by OpenGL ES, but rather will enter main() with undefined
    values."

Issues

    (1) How does this differ from OpenGL Shader Language support?

    RESOLVED. This is based on the language from the OpenGL Shading
    Language 4.50 specification. The only difference is that GLSL
    allows initializers on uniform variables, whereas ESSL does not.
    Also have added the statement clarifying that "All such initializers
    will have been executed before, or on entry to, main()".

    (2) How should these global non-constant initializers be implemented?

    RESOLVED. They operate as if they are executed at the beginning of
    the main() block before any other statements. That is:

        vec4 v = ...;  // "..." is a valid non-const initializer at this point

        void main()
        {
            statement1;
            statement2;
        }

    means

        vec4 v;

        void main()
        {
          v = ...;
          statement1;
          statement2;
        }

    For a more complex example:

        uniform int i;
        int a = i;

        void foo();
        void main() { foo(); }

        int b = 2 * a;

        void foo() { /* what's b? */ }

    the same rule applies. The point in time 'b' gets it's value
    is at the beginning of main(). There is no problem like "b isn't
    visible in main()". Rather, 'b' exists and is initialized on entry
    to main(), but is just not visible (per normal language rules).
    The full semantics of this are implemented in the glslang reference
    compiler.

Revision History
    Rev.    Date         Author         Changes
    ----  -----------    ------------   ---------------------------------
      1   10-Jun-2016    dkoch          Initial draft based on GLSL 4.50
      2   04-Jul-2016    dkoch          Final edits before publishing

