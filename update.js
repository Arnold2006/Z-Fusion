// Bootstrap only — pulls latest code then hands off to _update_steps.js.
// This file should rarely if ever need to change.
// All new nodes, deps, and update logic go in _update_steps.js.
module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    {
      method: "script.start",
      params: {
        uri: "update_deps.js"
      }
    }
  ]
}
