This is a Facial Expression Detection project using YoLo11

## Github Instruction
1) Create and switch to a new branch:
`git checkout -b <branch_name>`

2) Push a new branch to the server (remote):
`git push -u origin <branch_name>`

3) Switch between existing branches:
`git checkout <branch_name>`

4) List all branches (local and remote):
`git branch -a`

5) Fetch the latest branch info from the server:
`git fetch origin`

### The Correct Workflow
Work: You make your changes in the develop branch.

Commit: You save your work (git add . and git commit -m "your message").

Push develop: You push your develop branch to the server (git push origin develop).

Merge (via Pull Request/Merge Request): You go to your Git provider (GitHub, GitLab, etc.) and create a Pull Request (PR) to merge develop into main.

Review: You (or a teammate) review the code.

Merge: Once approved, the code is merged into main on the server.

Update Local: You switch to main locally and pull the new changes.