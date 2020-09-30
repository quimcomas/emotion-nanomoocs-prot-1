const express = require('express');
const Datastore = require('nedb');

const app= express();
const xpress = require('express');
const router = xpress.Router();
const bodyParser = require('body-parser');
app.use(bodyParser.json());
app.use(express.json());
const port = process.env.PORT
app.listen(port,()=>console.log(`listening at ${port}`))
app.use(express.static('public'));
app.use(bodyParser.json({limit: '10000mb'}));
app.use(bodyParser.urlencoded({limit: '10000mb', extended: true}));

const database = new Datastore('database.db');
database.loadDatabase();

app.route('/api')
    .get(function(req, res) {
        database.find({},(err, data)=> {res.json(data)})
    })
    .post(function(req, res) {
        const data = req.body;
        //res.json(data);
        const timestamp = Date.now();
        data.timestamp= timestamp;
        database.insert(data);
        console.log(req.body);
        res.json({
            status: 'success',
            timestamp: timestamp,
            prediction: data
        });
    })

