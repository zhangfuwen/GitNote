# QCOM_alpha_test

Name

    QCOM_alpha_test

Name Strings

    GL_QCOM_alpha_test

Contact

    Maurice Ribble (mribble 'at' qualcomm.com)

Contributors

    Benj Lipchak
    Maurice Ribble

Status

    Complete.

Version

    Last Modified Date: October 11, 2010
    Revision: #2

Number

    OpenGL ES Extension #89

Dependencies

    OpenGL ES 2.0 is required.

    This extension is written against the OpenGL ES 2.0 specification.

Overview

    This extension reintroduces the alpha test per-fragment operation
    from OpenGL ES 1.x.  Some hardware has a dedicated unit capable of
    performing this operation, and it can save ALU operations in the fragment
    shader by avoiding the conditional discard.

New Procedures and Functions

    void AlphaFuncQCOM(enum func, clampf ref);

New Tokens

    Accepted by the <cap> parameter of Enable and Disable, and IsEnabled, and by
    the <pname> parameter of GetBooleanv, GetIntegerv, and GetFloatv:

        ALPHA_TEST_QCOM                             0x0BC0

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, and
    GetFloatv:

        ALPHA_TEST_FUNC_QCOM                        0x0BC1
        ALPHA_TEST_REF_QCOM                         0x0BC2

    Accepted by the <func> parameter of AlphaFuncQCOM:

        NEVER
        LESS
        EQUAL
        LEQUAL
        GREATER
        NOTEQUAL
        GEQUAL
        ALWAYS

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations)

    Add a new section 4.1.4 - Alpha Test

        The alpha test discards a fragment conditional on the outcome of a
        comparison between the incoming fragment's alpha value and a constant
        value.  The comparison is enabled or disabled with the generic Enable
        and Disable commands using the symbolic constant ALPHA_TEST.  When
        disabled, it is as if the comparison always passes.  The test is
        controlled with

            void AlphaFuncQCOM(enum func, clampf ref);

        <func> is a symbolic constant indicating the alpha test function; <ref>
        is a reference value.  <ref> is clamped to lie in [0,1], and then
        converted to a fixed-point value according to the rules given for an A
        component in section 2.12.8.  For purposes of the alpha test, the
        fragment's alpha value is also rounded to the nearest integer.  The
        possible constants specifying the test function are NEVER, ALWAYS, LESS,
        LEQUAL, EQUAL, GEQUAL, GREATER, or NOTEQUAL, meaning pass the fragment
        never, always, if the fragment's alpha value is less than, less than or
        equal to, equal to, greater than or equal to, greater than, or not equal
        to the reference value, respectively.

        The required state consists of the floating-point reference value, an
        eight-valued integer indicating the comparison function, and a bit
        indicating if the comparison is enabled or disabled.  The initial state
        is for the reference value to be 0 and the function to be ALWAYS.
        Initially, the alpha test is disabled.

Errors

    None

New State

    (table 6.17, Pixel Operations) add the following entries:

                                           Initial
    Get Value            Type  Get Command  Value   Description          Section
    -------------------  ----  -----------  ------  -------------------  -------
    ALPHA_TEST_QCOM      B     IsEnabled    False   Alpha test enabled   4.1.4
    ALPHA_TEST_FUNC_QCOM Z8    GetIntegerv  ALWAYS  Alpha test function  4.1.4
    ALPHA_TEST_REF_QCOM  R+    GetFloatv    0       Alpha test reference 4.1.4
                                                    value

Revision History

    #01    11/04/2007    Benj Lipchak    Created based on ES 1.1 spec language.
    #02    10/11/2010    Maurice Ribble  Updated to be QCOM extension and fixed
                                         some wording.
