const server = require('../server');

module.exports = (req, res) => {
  // Vercel passes the request to the native Node server
  server.emit('request', req, res);
};
