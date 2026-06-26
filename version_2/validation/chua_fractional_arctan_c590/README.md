# Chua fractional arctan c590 validation

This directory promotes the c590 arctan candidate as radius-limited finite
hiddenness evidence, matching the same conservative convention used for other
methodology examples.

## Status

`hiddenness_supported_under_tested_local_radii_with_macro_radius_review`

The promoted local claim is limited to tested radii through `0.3`.
Those local radii contain `8400` finite probes around all equilibria
and `0` target contacts.

## Extended-radius audit

The extended macroscopic radii are retained as review evidence, not as a reason
to erase the local-radius promotion:

| Radius | Tests | Contacts | Decision |
| --- | ---: | ---: | --- |
| `1e-05` | 300 | 0 | `no_contact_detected` |
| `0.0001` | 600 | 0 | `no_contact_detected` |
| `0.001` | 900 | 0 | `no_contact_detected` |
| `0.01` | 1200 | 0 | `no_contact_detected` |
| `0.03` | 1500 | 0 | `no_contact_detected` |
| `0.1` | 1800 | 0 | `no_contact_detected` |
| `0.3` | 2100 | 0 | `no_contact_detected` |
| `1` | 2400 | 22 | `macro_radius_contact_detected` |
| `2` | 2700 | 588 | `macro_radius_contact_detected` |


Contacts occur only at radii `1, 2`.
The total extended-radius contact count is `610` out of
`5100` macro-radius probes.

## Boundary

This is finite deterministic neighborhood evidence under the recorded Caputo
ABM full-memory contract. It is not a filled-ball proof and not a global basin
proof. The Wu2023 bibliographic ADM lane remains separate and non-promoted as a
Caputo hiddenness validation.
