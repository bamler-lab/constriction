/** The "start"-symbol from main.rs. */
ENTRY(start)

OUTPUT_FORMAT("elf64-x86-64")
OUTPUT_ARCH("i386:x86-64")

SECTIONS {

    /* Link Address: 4MiB */
    . = 0x400000;
    . += SIZEOF_HEADERS;

    .text 0x400000 : ALIGN (4096)
    {
        *(.text .text.*)
    } : rx

    .rodata : ALIGN (4096)
    {
      *(.rodata .rodata.*)
    } : r

    .data : ALIGN (4096)
    {
      *(.data .data.*)
      *(COMMON)

      /* put .bss in .data */
      *(.bss .bss.*)
    } : rw

    /* Information for unwinding & backtraces */
    .eh_frame : ALIGN (4096)
    {
      *(.eh_frame*)
    }

}
