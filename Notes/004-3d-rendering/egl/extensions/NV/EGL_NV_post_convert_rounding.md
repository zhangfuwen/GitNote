# NV_post_convert_rounding

Name

    NV_post_convert_rounding

Name Strings

    EGL_NV_post_convert_rounding

Contributors

    Bryan Eyler, NVIDIA
    Daniel Kartch, NVIDIA
    Greg Roth, NVIDIA
    Mark Vojkovich, NVIDIA
    Nicolai de Haan Brogger, NVIDIA
    Peter Pipkorn, NVIDIA
    
Contacts

    Nicolai de Haan Brogger, NVIDIA Corporation (nicolaid 'at' nvidia.com)

Status

    Complete

Version

    Last Modified 17 Oct 2012
    Version 2

Number

    EGL Extension #44

Dependencies

    Requires EGL 1.0.

    This extension is written against the wording of the EGL 1.4
    Specification.

Overview

    This extension defines the conversions for posting operations
    when the destination's number of components or component sizes do
    not match the color buffer. This extension supports posting a 24 bit
    (888) color buffer to a 16 bit (565) destination buffer, posting a
    16 bit (565) color buffer to a 24 bit (888) destination buffer, and
    posting a component that is present in the source buffer, but not
    present in the destination buffer.    

New Procedures and Functions

    None

Changes to Chapter 3 of the EGL 1.4 Specification (EGL Functions and
Errors)

    In Chapter 3.9.3, replace paragraph 3 with:
        "    For each color component, if the bit depth of the color
         buffer being posted is 24 (888) and the destination buffer is
         16 (565), the lower order bits of the color buffer are
         truncated.

         If the bit depth of the destination buffer is 24 (888) and the
         color buffer being posted is 16 bit (565), a RGB gain and
         rounding operation is applied to the color buffer values prior
         to posting. The destination buffer will contain the rounded
         (nearest) and clamped result of the vector product of [1.03125,
         1.015625, 1.03125] with the RGB values of the color buffer.

         For cases where a component is present in the color buffer but
         the matching component is not present in the destination
         buffer, that component will be dropped."

    In Chapter 3.9.3, remove paragraph 5.

Issues

Revision History
#2 (Greg Roth, Oct 17, 2012)
   - Clarify limitations and reformat a bit.

#1 (Nicolai de Haan Brogger, July 07, 2010)
   - First Draft
