# WebApp protocol

_Version: 0.11 (025-09-2023)_

## Table of content

1.  [Terminology](protocol.md#terminology)
2.  [Overview](protocol.md#overview)
3.  [Session commands](protocol.md#session-commands)
    *   [Get a list of all active sessions](protocol.md#get-a-list-of-all-active-sessions)
    *   [Create a new session](protocol.md#create-a-new-session)
    *   [Remove an existing session](protocol.md#remove-an-existing-session)
    *   [Connect to an existing session](protocol.md#connect-to-an-existing-session)
    *   [Disconnect from an existing session](protocol.md#disconnect-from-an-existing-session)
    *   [Update all models of a session by importing data from an XML file](protocol.md#update-all-models-of-a-session-by-importing-data-from-an-xml-file)
    *   [Export all models of a session as XML file](protocol.md#export-all-models-of-a-session-as-xml-file)

4.  [Model instance commands](protocol.md#model-instance-commands)
    *   [Get a list of all model instances of a specific session](protocol.md#get-a-list-of-all-model-instances-of-a-specific-session)
    *   [Create a new model instance](protocol.md#create-a-new-model-instance)
    *   [Remove an existing model instance](protocol.md#remove-an-existing-model-instance)
    *   [Get all attributes of all components of a specific model](protocol.md#get-all-attributes-of-all-components-of-a-specific-model)
    *   [Get all attributes of a specific component of a specific instance](protocol.md#get-all-attributes-of-a-specific-component-of-a-specific-instance)
    *   [Get a specific attribute of a specific component of a specific instance](protocol.md#get-a-specific-attribute-of-a-specific-component-of-a-specific-instance)
    *   [Update a global attribute of a specific model instance](protocol.md#update-a-global-attribute-of-a-specific-model-instance)
    *   [Update a specific attribute of a specific component of a specific model instance](protocol.md#update-a-specific-attribute-of-a-specific-component-of-a-specific-model-instance)
    *   [Evaluate a specific component of a specific model instance](protocol.md#evaluate-a-specific-component-of-a-specific-model-instance)
    *   [Refine a specific model instance](protocol.md#refine-a-specific-model-instance)
    *   [Update a specific component of a specific model instance by importing data from an XML file](protocol.md#update-a-specific-component-of-a-specific-model-instance-by-importing-data-from-an-xml-file)
    *   [Export a specific model instance as XML file](protocol.md#export-a-specific-model-instance-as-xml-file)
    *   [Export a specific component of a specific model instance as XML file](protocol.md#export-a-specific-component-of-a-specific-model-instance-as-xml-file)

5.  [Models](protocol.md#models)
    *   [Global model attributes](protocol.md#global-model-attributes)
    *   [B-spline models](protocol.md#b-spline-models)
        *   [B-spline model options](protocol.md#b-spline-model-options)
        *   [B-spline model attributes](protocol.md#b-spline-model-attributes)

6.  [Descriptors](protocol.md#descriptors)
    *   [Option-type descriptor](protocol.md#option-type-descriptor)
    *   [Input/output-type descriptor](protocol.md#input%2Foutput-type-descriptor)
    *   [Capability descriptor](protocol.md#capability-descriptor)

## Terminology

- **Front-end** is the front-end application enabling interaction with the user. One or more front-end applications can be running at the same time on the same or on different devices and connecting to the same or different sessions running on the back-end.
- **Back-end** is the back-end application executing the requests. One or more back-end applications can be running at the same time on the same or on different devices provided they accept requests on different TCP port so that a client can connect to a specific back-end.
- **Session** is the 'scene' presented to one or more users. A back-end can run one or more sessions at the same time. Each session can be identified by its UUID. A front-end application can only connect to a single session at a time but multiple front-end applications can connect to the same session to enable collaborative design.
- **Model** is the blueprint of a particular type of physical model (e.g. Poisson's equation in 2d). It is realized as library that is loaded dynamically by the back-end application during startup. Each instance of a model blueprint can be customized (e.g., number of control knots per spatial direction) during its creation process. The list of available models is communicated when creating a new session or connecting to an existing one.
- **Instance** is a customized model instantiation running in a session. A session can contain one or more instances of the same or different models. Instances are identified by their number (integer value starting at 0 and being increased for each new instance that is created).
- **Component** is a part of a model, e.g., the geometry, the loadvector, or the solution field, that can be addressed individually. Components are separated into **inputs** and **outputs**. Inputs such as the geometry are components that can be modified by the user in the UI. Outputs such as the solution are fields that can be selected for visualization.

## Overview

All WebApps implement the following [WebSocket](https://en.wikipedia.org/wiki/WebSocket)-based client-server protocol. All communication is initiated by the client and responded by the server. Only broadcasts are initiated by the server and responded by all clients, e.g., when a client request leads to a state change of the server, then an update is broadcasted to all clients connected to the same session.

The format of the [JSON](https://en.wikipedia.org/wiki/JSON)-based protocol is as follows:

_Client request_

```json
{
    "id"      : <UUID>,
    "request" : <command>[/<session-id>][/<token0>]...[/<tokenN>],
  [ "data"    : {...} ]
}
```

_Server response_

```json
{
    "request" : <UUID>,
    "status"  : <integer>,
  [ "reason"  : <string> ],
  [ "data"    : {...} ]
}
```

*   `UUID` is chosen by the client for each new request and copied by the server in the reply.

*   `request` encodes the actual `command`, optionally followed by one or more `tokens`

*   `status` is an integer from the following list of predefined integers:
    -  `0` : `success`
    -  `1` : `invalidRequest`
    -  `2` : `invalidCreateRequest`
    -  `3` : `invalidRemoveRequest`
    -  `4` : `invalidConnectRequest`
    -  `5` : `invalidDisconnectRequest`
    -  `6` : `invalidGetRequest`
    -  `7` : `invalidPutRequest`
    -  `8` : `invalidEvalRequest`
    -  `9` : `invalidRefineRequest`
    - `10` : `invalidLoadRequest`
    - `11` : `invalidSaveRequest`
    - `12` : `invalidImportRequest`
    - `13` : `invalidExportRequest`

*   `reason` is an optional string that contains human-readable information about a failed request. Successful requests with `status : 0` will not contain a `reason` field

*    `data` is an optional request/response-dependent payload

In what follows, only the non-generic parts of the protocol are specified in more detail.

## Session commands

### Get a list of all active sessions

_Client request_
```json
"request" : "get"
```

_Server response_
```json
"status"  : success (0)
"data"    : { "ids" : [<comma-separated list of session ids>] }
```
or
```json
"status"  : invalidGetRequest (6)
"reason"  : <string>
```

### Create a new session

_Client request_
```json
"request" : "create/session"
```

_Server response_
```json
"status"  : success (0)
"data"    : { "id" : session-id,
               "models" : [<comma-separated list of models>] }
```
or
```json
"status"  : invalidCreateRequest (2)
"reason"  : <string>
```
-   The `session-id` is generated by the server and serves as authentification mechanism.
-   The `models` list describes the models supported by the server.

      -   Each list entry has the form
          ```json
          { "name"        : <string>,
             "description" : <string>,
             "options"     : [<comma-separated list of options>],
             "inputs"      : [<comma-separated list of inputs>],
             "outputs"     : [<comma-separated list of outputs>] }
          ```
          The model's `name` is used for creating a specific type.
      -   Each entry in the `options` list has the form
          ```json
          { "name"        : <string>,
             "description" : <string>,
             "type"        : <optiontype descriptor>,
             "value"       : <values>,
             "default"     : <default value>
             "uiid"        : <integer specifying the position in the UI> }
          ```
          This information can be used to generate UI elements dynamically based on the capabilities of the server. See [`optiontype descriptor`](protocol.md#optiontype-descriptor) for details.

          _Example:_
          ```json
          { "name"        : "ncoeffs",
             "description" : "Number of coefficients",
             "type"        : [int,int],
             "value"       : [5,5],
             "default"     : [5,5],
             "uiid"        : 0 }
          ```
          or
          ```json
          { "name"        : "init",
             "description" : "Initialization of the coefficients",
             "type"        : "select",
             "value"       : ["zeros", "ones", "linear", "random", "greville"],
             "default"     : 2,
             "uiid"        : 1 }
          ```

      -   Each entry in the `inputs` and `outputs` lists has the form
          ```json
          { "name"        : <string>,
             "description" : <string>,
             "type"        : <iotype descriptor> }
          ```
          whereby the `iotype descriptor` must be one of the options given in

### Remove an existing session

_Client request_
```json
"request" : "remove/<session-id>"
```

_Server response_
```json
"status"  : success (0)
```
or
```json
"status"  : invalidRemoveRequest (3)
"reason"  : <string>
```

### Connect to an existing session

_Client request_
```json
"request" : "connect/<session-id>"
```

_Server response_
```json
"status"  : success (0)
```
or
```json
"status"  : invalidConnectRequest (4)
"reason"  : <string>
```

### Disconnect from an existing session

_Client request_
```json
"request" : "disconnect/<session-id>"
```

_Server response_
```json
"status"  : success (0)
```
or
```json
"status"  : invalidDisconnectRequest (5)
"reason"  : <string>
```

### Update all models of a session by importing data from an XML file

_Client request_
```json
{
   "id"      : <UUID>,
   "request" : importxml/<session-id>,
   "data"    : { "xml" : <xml string> }
}
```

_Server response_
```json
"status"  : success (0)
```
or
```json
"status"  : invalidLoadRequest (10)
"reason"  : <string>
```

Note that the session must have all model instances created in a form compatible with the XML file. That is, no new models are created from the XML file but only the attributes of existing models are updated by the data read from the XML file. If the models are not compatible with the data in the XML file the request terminates with an `invalidLoadRequest` status.

_Example:_

An example of an XML file for an entire session might look as follows:
```xml
<?xml version="1.0"?>
<xml>
   <Geometry type="BSpline" id="0" label="geometry">
      <Basis type="BSplineBasis">
         <KnotVector degree="1">0.000000 0.000000 0.333333 0.666667 1.000000 1.000000</KnotVector>
      </Basis>
      <coefs geoDim="3">0.000000 1.000000 1.000000 0.333333 1.000000 1.000000 0.666667 1.000000 1.000000 1.000000 1.000000 1.000000 </coefs>
   </Geometry>
   <Geometry type="BSpline" id="0" label="solution">
      <Basis type="BSplineBasis">
         <KnotVector degree="1">0.000000 0.000000 0.333333 0.666667 1.000000 1.000000</KnotVector>
      </Basis>
      <coefs geoDim="3">0.000000 0.000000 0.000000 0.866025 0.000000 0.000000 0.866025 0.000000 0.000000 0.000000 0.000000 0.000000 </coefs>
   </Geometry>
   <Geometry type="TensorBSpline2" id="1" label="geometry">
      <Basis type="TensorBSplineBasis2">
         <Basis type="BSplineBasis" index="0">
            <KnotVector degree="1">0.000000 0.000000 0.333333 0.666667 1.000000 1.000000</KnotVector>
         </Basis>
         <Basis type="BSplineBasis" index="1">
            <KnotVector degree="1">0.000000 0.000000 0.333333 0.666667 1.000000 1.000000</KnotVector>
         </Basis>
      </Basis>
      <coefs geoDim="3">0.000000 0.000000 1.000000 0.333333 0.000000 1.000000 0.666667 0.000000 1.000000 1.000000 0.000000 1.000000 0.000000 0.333333 1.000000 0.333333 0.333333 1.000000 0.666667 0.333333 1.000000 1.000000 0.333333 1.000000 0.000000 0.666667 1.000000 0.333333 0.666667 1.000000 0.666667 0.666667 1.000000 1.000000 0.666667 1.000000 0.000000 1.000000 1.000000 0.333333 1.000000 1.000000 0.666667 1.000000 1.000000 1.000000 1.000000 1.000000 </coefs>
   </Geometry>
   <Geometry type="TensorBSpline2" id="1" label="solution">
      <Basis type="TensorBSplineBasis2">
         <Basis type="BSplineBasis" index="0">
            <KnotVector degree="1">0.000000 0.000000 0.333333 0.666667 1.000000 1.000000</KnotVector>
         </Basis>
         <Basis type="BSplineBasis" index="1">
            <KnotVector degree="1">0.000000 0.000000 0.333333 0.666667 1.000000 1.000000</KnotVector>
         </Basis>
      </Basis>
      <coefs geoDim="3">0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.750000 0.000000 0.000000 0.750000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.750000 0.000000 0.000000 0.750000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 </coefs>
   </Geometry>
   <Geometry type="TensorBSpline3" id="2" label="geometry">
      <Basis type="TensorBSplineBasis3">
         <Basis type="BSplineBasis" index="0">
            <KnotVector degree="1">0.000000 0.000000 0.333333 0.666667 1.000000 1.000000</KnotVector>
         </Basis>
         <Basis type="BSplineBasis" index="1">
            <KnotVector degree="1">0.000000 0.000000 0.333333 0.666667 1.000000 1.000000</KnotVector>
         </Basis>
         <Basis type="BSplineBasis" index="2">
            <KnotVector degree="1">0.000000 0.000000 0.333333 0.666667 1.000000 1.000000</KnotVector>
         </Basis>
      </Basis>
      <coefs geoDim="3">0.000000 0.000000 0.000000 0.333333 0.000000 0.000000 0.666667 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.333333 0.000000 0.333333 0.333333 0.000000 0.666667 0.333333 0.000000 1.000000 0.333333 0.000000 0.000000 0.666667 0.000000 0.333333 0.666667 0.000000 0.666667 0.666667 0.000000 1.000000 0.666667 0.000000 0.000000 1.000000 0.000000 0.333333 1.000000 0.000000 0.666667 1.000000 0.000000 1.000000 1.000000 0.000000 0.000000 0.000000 0.333333 0.333333 0.000000 0.333333 0.666667 0.000000 0.333333 1.000000 0.000000 0.333333 0.000000 0.333333 0.333333 0.333333 0.333333 0.333333 0.666667 0.333333 0.333333 1.000000 0.333333 0.333333 0.000000 0.666667 0.333333 0.333333 0.666667 0.333333 0.666667 0.666667 0.333333 1.000000 0.666667 0.333333 0.000000 1.000000 0.333333 0.333333 1.000000 0.333333 0.666667 1.000000 0.333333 1.000000 1.000000 0.333333 0.000000 0.000000 0.666667 0.333333 0.000000 0.666667 0.666667 0.000000 0.666667 1.000000 0.000000 0.666667 0.000000 0.333333 0.666667 0.333333 0.333333 0.666667 0.666667 0.333333 0.666667 1.000000 0.333333 0.666667 0.000000 0.666667 0.666667 0.333333 0.666667 0.666667 0.666667 0.666667 0.666667 1.000000 0.666667 0.666667 0.000000 1.000000 0.666667 0.333333 1.000000 0.666667 0.666667 1.000000 0.666667 1.000000 1.000000 0.666667 0.000000 0.000000 1.000000 0.333333 0.000000 1.000000 0.666667 0.000000 1.000000 1.000000 0.000000 1.000000 0.000000 0.333333 1.000000 0.333333 0.333333 1.000000 0.666667 0.333333 1.000000 1.000000 0.333333 1.000000 0.000000 0.666667 1.000000 0.333333 0.666667 1.000000 0.666667 0.666667 1.000000 1.000000 0.666667 1.000000 0.000000 1.000000 1.000000 0.333333 1.000000 1.000000 0.666667 1.000000 1.000000 1.000000 1.000000 1.000000 </coefs>
   </Geometry>
   <Geometry type="TensorBSpline3" id="2" label="solution">
      <Basis type="TensorBSplineBasis3">
         <Basis type="BSplineBasis" index="0">
            <KnotVector degree="1">0.000000 0.000000 0.333333 0.666667 1.000000 1.000000</KnotVector>
         </Basis>
         <Basis type="BSplineBasis" index="1">
            <KnotVector degree="1">0.000000 0.000000 0.333333 0.666667 1.000000 1.000000</KnotVector>
         </Basis>
         <Basis type="BSplineBasis" index="2">
            <KnotVector degree="1">0.000000 0.000000 0.333333 0.666667 1.000000 1.000000</KnotVector>
         </Basis>
      </Basis>
      <coefs geoDim="3">0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.649519 0.000000 0.000000 0.649519 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.649519 0.000000 0.000000 0.649519 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.649519 0.000000 0.000000 0.649519 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.649519 0.000000 0.000000 0.649519 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 </coefs>
   </Geometry>
</xml>
```
Here, the first model (`id=0`) of the session has to be of type `BSplineCurve`, the second model (`id=1`) has to be of type `BSplineSurface`, and the third model (`id=2`) has to be of type `BSplineVolume`. The XML format is adopted from [G+Smo](https://gismo.github.io/Tutorial01.html#xml01).

### Export all models of a session as XML file

_Client request_
```json
{
   "id"      : <UUID>,
   "request" : exportxml/<session-id>,
}
```

_Server response_
```json
"status"  : success (0)
"data"    : { "xml" : <xml string> }
```
or
```json
"status"  : invalidSaveRequest (11)
"reason"  : <string>
```

## Model instance commands

### Get a list of all model instances of a specific session

_Client request_
```json
"request" : "get/<session-id>"
```

_Server response_
```json
"status"  : success (0)
"data"    : { "ids" : [<comma-separated list of model instances>] }
```
or
```json
"status"  : invalidGetRequest (6)
"reason"  : <string>
```

### Create a new model instance

_Client request_
```json
"request" : "create/<session-id>/<model-type>"
"data"    : {...}
```

The list of supported models is sent by the server upon creating a new session or connecting to an existing session. See [Create a new session](protocol.md#create-a-new-session) and [Models](protocol.md#models) for details.

_Server response_
```json
"status"  : success (0)
"data"    : { "id" : instance }
```
or
```json
"status"  : invalidCreateRequest (2)
"reason"  : <string>
```

### Remove an existing model instance

_Client request_
```json
"request" : "remove/<session-id>/<instance>"
```

_Server response_
```json
"status"  : success (0)
```
or
```json
"status"  : invalidRemoveRequest (3)
"reason"  : <string>
```

### Get all attributes of all components of a specific model

_Client request_
```json
"request" : "get/<session-id>/<instance>"
```

The attributes for the different models are given in [Models](protocol.md#models).

_Server response_
```json
"status"  : success (0)
"data"    : {...}
```
or
```json
"status"  : invalidGetRequest (6)
"reason"  : <string>
```

### Get all attributes of a specific component of a specific instance

_Client request_
```json
"request" : "get/<session-id>/<instance>/<component>"
```

The attributes for the different models are given in [Models](protocol.md#models).

_Server response_
```json
"status"  : success (0)
"data"    : {...}
```
or
```json
"status"  : invalidGetRequest (6)
"reason"  : <string>
```

### Get a specific attribute of a specific component of a specific instance

_Client request_
```json
"request" : "get/<session-id>/<instance>/<component>/<attribute>"
```

The attributes for the different models are given in [Models](protocol.md#models).

_Server response_
```json
"status"  : success (0)
"data"    : {...}
```
or
```json
"status"  : invalidGetRequest (6)
"reason"  : <string>
```
The `data` will be formatted in the same format as in the _get all attributes_ case.

### Update a global attribute of a specific model instance

_Client request_
```json
"request" : "put/<session-id>/<instance>/<global attribute>
```

The updatable _global_ attributes for the different models are given in [Models](protocol.md#models).

_Server response_
```json
"status"  : success (0)
"data"    : {...}
```
or
```json
"status"  : invalidPutRequest (7)
"reason"  : <string>
```

### Update a specific attribute of a specific component of a specific model instance

_Client request_
```json
"request" : "put/<session-id>/<instance>/<component>/<attribute>"
```

The updatable attributes for the different models are given in [Models](protocol.md#models).

_Server response_
```json
"status"  : success (0)
"data"    : {...}
```
or
```json
"status"  : invalidPutRequest (7)
"reason"  : <string>
```

### Evaluate a specific component of a specific model instance

_Client request_
```json
"request" : "eval/<session-id>/<instance>/<component>"
"data"    : {
               "resolution" : [<list of integers]
            }
```
The `component` must be one of the model's outputs (i.e. the `name` attribute) defined during [session creation](protocol.md#Create-a-new-session).

If the optional `data` and `resolution` are not present, the default resolution is 25 in each parametric dimension.

_Server response_
```json
"status"  : success (0)
"data" : {
            "values" : [<list of floats in lexicographical order>]
         }
```
or
```json
"status"  : invalidEvalRequest (8)
"reason"  : <string>
```

### Refine a specific model instance

_Client request_
```json
"request" : "refine/<session-id>/<instance>"
"data"    : {
               [ "num" : <integer> (default value is 1) ]
               [ "dim" : <integer> (default value is -1) ]
            }
```
If no `data` field is provided the default values are adopted.

_Server response_
```json
"status"  : success (0)
```
or
```json
"status"  : invalidRefineRequest (9)
"reason"  : <string>
```

### Update a specific model instance by importing data from an XML file

_Client request_
```json
{
   "id"      : <UUID>,
   "request" : importxml/<session-id>/<instance>,
   "data"    : { "xml" : <xml string> }
}
```

_Server response_
```json
"status"  : success (0)
```
or
```json
"status"  : invalidLoadRequest (10)
"reason"  : <string>
```

Note that the model must exist in the session in a form compatible with the XML file. If the model is not compatible with the data in the XML file the request terminates with an `invalidLoadRequest` status.

_Example:_

An example of an XML file for a model instance might look as follows:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<xml>
   <Geometry type="TensorBSpline2" id="0" label="geometry">
      <Basis type="TensorBSplineBasis2">
         <Basis type="BSplineBasis" index="0">
            <KnotVector degree="1">0.000000 0.000000 0.500000 1.000000 1.000000</KnotVector>
         </Basis>
         <Basis type="BSplineBasis" index="1">
            <KnotVector degree="1">0.000000 0.000000 1.000000 1.000000</KnotVector>
         </Basis>
      </Basis>
      <coefs geoDim="3">0.000000 0.000000 1.000000 0.500000 0.000000 1.000000 1.000000 0.000000 1.000000 0.000000 1.000000 1.000000 0.500000 1.000000 1.000000 1.000000 1.000000 1.000000 </coefs>
   </Geometry>
   <Geometry type="TensorBSpline2" id="0" label="solution">
      <Basis type="TensorBSplineBasis2">
         <Basis type="BSplineBasis" index="0">
            <KnotVector degree="1">0.000000 0.000000 0.500000 1.000000 1.000000</KnotVector>
         </Basis>
         <Basis type="BSplineBasis" index="1">
            <KnotVector degree="1">0.000000 0.000000 1.000000 1.000000</KnotVector>
         </Basis>
      </Basis>
      <coefs geoDim="3">0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 </coefs>
   </Geometry>
</xml>
```

### Update a specific component of a specific model instance by importing data from an XML file

_Client request_
```json
{
   "id"      : <UUID>,
   "request" : importxml/<session-id>/<instance>/<component>
   "data"    : { "xml" : <xml string> }
}
```

_Server response_
```json
"status"  : success (0)
```
or
```json
"status"  : invalidLoadRequest (10)
"reason"  : <string>
```

Note that the model must exist in the session in a form compatible with the XML file. If the model is not compatible with the data in the XML file the request terminates with an `invalidLoadRequest` status.

_Example:_

An example of an XML file for a model instance might look as follows:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<xml>
   <Geometry type="TensorBSpline2" id="0" label="geometry">
      <Basis type="TensorBSplineBasis2">
         <Basis type="BSplineBasis" index="0">
            <KnotVector degree="1">0.000000 0.000000 0.500000 1.000000 1.000000</KnotVector>
         </Basis>
         <Basis type="BSplineBasis" index="1">
            <KnotVector degree="1">0.000000 0.000000 1.000000 1.000000</KnotVector>
         </Basis>
      </Basis>
      <coefs geoDim="3">0.000000 0.000000 1.000000 0.500000 0.000000 1.000000 1.000000 0.000000 1.000000 0.000000 1.000000 1.000000 0.500000 1.000000 1.000000 1.000000 1.000000 1.000000 </coefs>
   </Geometry>
</xml>
```

### Export a specific model instance as XML file

_Client request_
```json
{
   "id"      : <UUID>,
   "request" : exportxml/<session-id>/<instance>
}
```

_Server response_
```json
"status"  : success (0)
"data"    : { "xml" : <xml string> }
```
or
```json
"status"  : invalidSaveRequest (11)
"reason"  : <string>
```

### Export a specific component of a specific model instance as XML file

_Client request_
```json
{
   "id"      : <UUID>,
   "request" : exportxml/<session-id>/<instance>/<component>
}
```

_Server response_
```json
"status"  : success (0)
"data"    : { "xml" : <xml string> }
```
or
```json
"status"  : invalidSaveRequest (11)
"reason"  : <string>
```

## Models

### Global model attributes

All models support the following global attributes:

#### Transform

The [transformation matrix](https://en.wikipedia.org/wiki/Transformation_matrix) is a [4x4 matrix](https://threejs.org/docs/#api/en/math/Matrix4) from which the global translation and rotation of the model can be computed.

Each model stores this matrix internally and returns its values upon request:

_Client request_
```json
"request" : "get/<session-id>/<instance>/transform"
```

_Server response_
```json
"status" : success(0),
"data"   : {
              "elements" : [<comma-separated list of floats>]
           }
```

The model's internal transformation matrix can be updated by the following request:
_Client request_
```json
"request" : "put/<session-id>/<instance>/transform",
"data"    : {
               "elements" : [<comma-separated list of floats>]
            }
```

_Server response_
```json
"status" : success(0)
```

### B-spline models

The following B-spline models are implemented:
-    `BSplineCurve` : uni-variate B-spline curve
-    `BSplineSurface` : bi-variate B-spline surface
-    `BSplineVolume` : tri-variate B-spline volume

#### B-spline model options

The following `data` can be passed during [model creation](protocol.md#Create-a-new-model):

```json
"data" : {
            "degree"     : <integer> valid values are
                                     0 constant,
                                     1 linear (default),
                                     2 quadratic,
                                     3 cubic,
                                     4 quartic,
                                     5 quintic
            "init"       : <integer> valid values are
                                     0 zeros,
                                     1 ones,
                                     2 linear,
                                     3 random,
                                     4 Greville (default)
            "ncoeffs"    : [<comma-separated list of integers>],
                                     [4,...] (default)
            "nonuniform" : <bool>    valid values are
                                     0 uniform (default),
                                     1 nonuniform
         }
```

#### B-spline model attributes

The following `data` can be passed when [getting all model attributes](protocol.md#Get-all-attributes-of-a-specific-model) or [getting a specific model attribute](protocol.md#Get-a-specific-attribute-of-a-specific-model).
```json
"data" : {
            "geoDim"  : <integer>    valid values are
                                     1 univariate,
                                     2 bivariate,
                                     3 trivariate,
                                     4 quadrupelvariate
            "parDim"  : <integer>    valid values are
                                     1 univariate,
                                     2 bivariate,
                                     3 trivariate,
                                     4 quadrupelvariate
            "degrees" : [<comma-separated list of integers>],
            "ncoeffs" : [<comma-separated list of integers>],
            "nknots"  : [<comma-separated list of integers>],
            "coeffs"  : [[comma-separated list of lists of floats]],
            "knots"   : [[comma-separated list of lists of floats]]
         }
```

-   The attribute `coeffs` is stored as a list of lists with the inner list containing all coefficients in lexicographical order(i.e. with the first parametric dimension $\xi_1$ running quickest and the last parametric dimension $\xi_{\text{par}_\text{dim}}$ running slowest) and the outer list containing the coefficients of the different geometric dimensions.

    __Example:__
    ```json
    "coeffs" : [[x_1_1, x_2_1, ..., x_ncoeffs(1)_ncoeffs(2)], ...,
                [z_1_1, z_1_2, ..., z_ncoeffs(1)_ncoeffs(2)]]
    ```
    for a bivariate parametrization with coefficients in $\mathbb{R}^3$.

    Currently, the float values are stored as plain JSON string but it is planned to change to binary base64 encoding in the future.

-   The attribute `coeffs` can be modifief by the `put` command.

    __Example:__
    ```json
    "request" : put/<session-id>/<instance>/coeffs
    "data"    : {
                  "indices" : [0, 6, 9],
                  "coeffs"  : [[0.5, 0.2], [0.3, 0.6], [0.9, 1.2]]
                }
    ```
    This will update the coordinates of the two-dimensional coefficients with global indices `0`, `6`, and `9` to the values `[0.5, 0.2]`, `[0.3, 0.6]`, and `[0.9, 1.2]`. It is not assumed that the indices and coordinates are numbered.

-   The attribute `knots` is stored as a list of lists with the inner list containing the univariate knot vectors and the outer list containing the different parametric dimensions.

    __Example:__
    ```json
    "knots" : [[k1_1, ..., k1_nknots[0]], ...,
               [kparDim_1, ..., kparDim_nknots[parDim]]]
    ```

    Currently, the float values are stored as plain JSON string but it is planned to change to binary base64 encoding in the future.

## Descriptors

### Option-type descriptor

The `optiontype descriptor` must be one of the following

| `type`        | description  | value format example | enum value |
|--------------:|:-------------|:---------------------|------------|
| `int`         | integer value| `"value" : 5` or     |
||| `"value" : [5,5]`  |
| `float`       | float value  | `"value" : 5.0` or  |
||| `"value" : [5.0,5.0]` |
| `text`      | string value | `"value" : "string"` or |
||| `"value" : ["string1","string2"]` |
| `list`    | can select no, one or multiple options | `"value" : { "option0", "option1" }`|
| `select` | must select one option  | `"value" : { "option0", "option1" }`|

Types `list` and `select` correspond to enumerators with the mapping `option0 -> 0`, `option1 -> 1` etc. It is therefore expected that if, say, `option1` is chosen in a `select` type, the UI sends the value `1` in the next request. Similarly, if, say, `options0` and `options1` are chosen in a `list` type, the UI sends the value `[0,1]` in the next request.

### Input/output-type descriptor

The `iotype descriptor` must be one of the following

 | `type`        | description   | enum value |
 |--------------:|:--------------|------------|
 | `scalar`      | scalar value (e.g., error or drag/lift coefficient) | 0 |
 | `scalarfield` | scalar field (e.g., pressure) | 1 |
 | `vectorfield` | vector field (e.g., velocity) | 2 |
 | `scalarfield_boundary` | scalar field at the boundary (e.g., pressure boundary conditions) | 3 |
 | `vectorfield_boundary` | vector field at the boundary (e.g., velocity) | 4 |

 ### Capability descriptor

 The `capability descriptor` must be one of the following

 | `type`        | description   | enum value |
 |--------------:|:--------------|------------|
 | `create`      | create model | 0 |
 | `remove`      | remove model | 1 |
 | `parameters`  | model has extra parameters |2 |
 | `eval`        | evaluate model | 3 |
 | `refine`      | h-refine model  | 4 |
 | `elevate`     | p-refine model (degree elevation) | 5 |
 | `increase`    | p-refine model (degree increase) | 6 |
 | `reparameterize` | reparameterize the model's geometry | 7 |
 | `load`        | load model from file | 8 |
 | `save`        | save model to file   | 9 |
 | `importXML`   | import object from XML file | 10 |
 | `exportXML`   | export object to XML file   | 11 |
 | `computeL1error` | compute L1 error of the model | 12 |
 | `computeL2error` | compute L2 error of the model | 13 |
 | `computeH1error` | compute H1 error of the model | 14 |
---
