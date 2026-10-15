#include "walker_frame.h"

#include <string.h>

uint16_t Walker_Crc16(const uint8_t *data, size_t length)
{
  uint16_t crc = 0xFFFFU;
  for (size_t i = 0U; i < length; i++)
  {
    crc ^= (uint16_t)((uint16_t)data[i] << 8);
    for (unsigned bit = 0U; bit < 8U; bit++)
      crc = (crc & 0x8000U) ? (uint16_t)((crc << 1) ^ 0x1021U) : (uint16_t)(crc << 1);
  }
  return crc;
}

void WalkerParser_Init(WalkerParser *parser)
{
  parser->len = 0U;
}

/* After a bad frame, keep any later 0xAA 0x55 that started inside it, so one corrupt frame
 * cannot swallow the good frame that follows. */
static void Resync(WalkerParser *parser)
{
  uint8_t i;
  for (i = 1U; i < parser->len; i++)
  {
    if (parser->buf[i] == 0xAAU)
    {
      uint8_t keep = (uint8_t)(parser->len - i);
      memmove(parser->buf, &parser->buf[i], keep);
      parser->len = keep;
      return;
    }
  }
  parser->len = 0U;
}

int WalkerParser_Push(WalkerParser *parser, uint8_t byte, WalkerFrame *out, uint32_t *bad_crc)
{
  if (parser->len == 0U)
  {
    if (byte == 0xAAU) parser->buf[parser->len++] = byte;
    return 0;
  }
  if (parser->len == 1U)
  {
    if (byte == 0x55U) parser->buf[parser->len++] = byte;
    else parser->len = (byte == 0xAAU) ? 1U : 0U; /* buf[0] is already 0xAA */
    return 0;
  }
  parser->buf[parser->len++] = byte;
  if (parser->len < WALKER_FRAME_LEN) return 0;

  const uint16_t expected = (uint16_t)(parser->buf[WALKER_FRAME_LEN - 2U] |
                                       ((uint16_t)parser->buf[WALKER_FRAME_LEN - 1U] << 8));
  if (Walker_Crc16(&parser->buf[2], WALKER_FRAME_LEN - 4U) != expected)
  {
    if (bad_crc != NULL) (*bad_crc)++;
    Resync(parser);
    return 0;
  }
  out->seq = parser->buf[2];
  out->flags = parser->buf[3];
  for (unsigned i = 0U; i < WALKER_JOINTS; i++)
    out->pulse_us[i] = (uint16_t)(parser->buf[4U + 2U * i] | ((uint16_t)parser->buf[5U + 2U * i] << 8));
  parser->len = 0U;
  return 1;
}
