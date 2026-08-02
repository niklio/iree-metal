# Security policy

iree-metal developer previews are experimental and are not supported for
production or untrusted-input isolation.

Please report suspected vulnerabilities privately with GitHub's
**Report a vulnerability** feature in the Security tab of
`niklio/iree-metal`. Include the affected release, impact, reproduction steps,
and any proposed mitigation. Do not open a public issue before the maintainer
has had a reasonable opportunity to assess the report.

Only the newest developer preview receives security fixes. Because previews
have no stability commitment, a fix may require upgrading to a new profile or
wheel pair. Published wheels should be verified against `SHA256SUMS` and, when
available, the GitHub artifact attestation.
