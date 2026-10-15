/* Laptop -> STM32 servo command frame parser. Plain C (no HAL) so it can be unit-tested on a PC.
 *
 * Frame, little-endian, 44 bytes:
 *   0xAA 0x55 | seq:u8 | flags:u8 | 19 x u16 pulse_us | crc16:u16
 *   flags bit0 = RELAX
 *   crc16 = CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over bytes [2..41] (seq .. last pulse)
 * Pulses are in SIMULATOR joint order (see walker_link.c for the mapping to PCA channels).
 */
#ifndef WALKER_FRAME_H
#define WALKER_FRAME_H

#include <stddef.h>
#include <stdint.h>

#define WALKER_JOINTS     19U
#define WALKER_FRAME_LEN  44U
#define WALKER_FLAG_RELAX 0x01U

typedef struct
{
  uint8_t buf[WALKER_FRAME_LEN];
  uint8_t len;
} WalkerParser;

typedef struct
{
  uint8_t seq;
  uint8_t flags;
  uint16_t pulse_us[WALKER_JOINTS];
} WalkerFrame;

uint16_t Walker_Crc16(const uint8_t *data, size_t length);
void WalkerParser_Init(WalkerParser *parser);

/* Feed one byte. Returns 1 when a CRC-valid frame completed (written to *out), else 0.
 * *bad_crc is incremented for every complete-length frame that fails the CRC. */
int WalkerParser_Push(WalkerParser *parser, uint8_t byte, WalkerFrame *out, uint32_t *bad_crc);

#endif
