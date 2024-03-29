# ARM_mali_shader_binary

Name

    ARM_mali_shader_binary

Name Strings

    GL_ARM_mali_shader_binary

Contributors

    Aske Simon Christensen, ARM
    Erik Faye-Lund, ARM
    Bruce Merry, ARM

Contact

    Jan-Harald Fredriksen (jan-harald.fredriksen 'at' arm.com)

Status

    Shipping

Version

    Last Modified Date: January 5, 2011

Number

    OpenGL ES Extension #81

Dependencies

    OpenGL ES 2.0 is required.

    Written based on the wording of the OpenGL ES 2.0 specification.

Overview

    This extension enables OpenGL ES 2.0 applications running on ARM
    Mali graphics cores to use shaders precompiled with the Mali ESSL
    shader compiler.

    The shader objects loaded with this extension are equivalent to
    shaders created from source, i.e. there are no additional
    restrictions on which other shader objects they can be linked to,
    nor on which OpenGL ES states they can be used with.

Issues

    None

New Procedures and Functions

    None

New Tokens

    Accepted by the <binaryFormat> parameter of ShaderBinary:

        MALI_SHADER_BINARY_ARM              0x8F60

Additions to Chapter 2 of the OpenGL ES 2.0 Specification

    At the end of section 2.10.2 (Loading Shader Binaries), add:

        "Any shader object passed to the ShaderBinary function with
        a <binaryFormat> of MALI_SHADER_BINARY_ARM will have its information
        log overwritten with information about the loading process."

Errors

    An INVALID_VALUE error is generated if the <binary> parameter points
    to an invalid binary stream that is either not appropriate for the
    core version (or core revision) or produced by an incompatible or
    outdated version of the Mali ESSL compiler or with inappropriate
    compiler options.

New State

    None

New Implementation Dependent State

    None


Revision History

    #1  08/27/2008   Erik Faye-Lund           First draft.
    #2  09/04/2008   Aske Simon Christensen   Actual enum value.
                                              Some adjustments.
                                              Mention shader info log.
    #3  09/05/2008   Aske Simon Christensen   Error and log behavior.
    #4  15/07/2010   Bruce Merry              Change status.
                                              Change contact.
                                              Clarify that shader log is
                                              only overwritten when using
                                              MALI_SHADER_BINARY_FORMAT_ARM.
    #5  05/01/2011   Jan-Harald Fredriksen    Fixed typos.
