# Artifact Appendix

Paper title: **TimberStrike: Dataset Reconstruction Attack Revealing Privacy Leakage in Federated Tree-Based Systems**

Artifacts HotCRP Id: **#12**

Requested Badge: **Available**

## Description

This artifact contains the complete source code that implements the dataset reconstruction attack described in the paper "TimberStrike: Dataset Reconstruction Attack Revealing Privacy Leakage in Federated Tree-Based Systems". It includes all components required to reproduce the experimental pipeline, including the attack implementation, experimental setup, and evaluation scripts.

The artifact is tightly coupled with the paper, as it allows reviewers and readers to validate the core contribution — namely, the ability to reconstruct private training data in federated learning systems employing decision-tree-based models. The implementation follows the methodology described in Section 5 of the paper and supports the reproduction of all experimental results presented in Section 6.

### Security/Privacy Issues and Ethical Concerns

This artifact does not pose any security or privacy risk to the reviewer's machine. It does not contain malware samples, nor does it require the disabling of security mechanisms such as firewalls, ASLR, or antivirus systems.

From an ethical standpoint, this research was conducted with a strong commitment to responsible disclosure and the broader security of federated learning systems. While the artifact demonstrates a privacy attack, the intent and presentation are in line with established ethical principles. We openly discuss the ethical implications of our work in the paper (see Appendix A of the paper), emphasizing the importance of transparency and the need to empower users and system designers to understand and mitigate emerging threats.

## Environment

### Accessibility

The artifact is publicly available at the following GitHub repository:

> [https://github.com/necst/TimberStrike](https://github.com/necst/TimberStrike)
> Commit evaluated: [301bf6c4cccd7d9222af969fc792c88612d2c848](https://github.com/necst/TimberStrike/commit/301bf6c4cccd7d9222af969fc792c88612d2c848)

If the repository is updated in response to reviewer feedback, a new commit ID or tag will be provided in the comment to maintain traceability.
