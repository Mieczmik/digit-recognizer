??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.22unknown8??
?
'Adam/module_wrapper_25/conv2d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'Adam/module_wrapper_25/conv2d_17/bias/v
?
;Adam/module_wrapper_25/conv2d_17/bias/v/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_25/conv2d_17/bias/v*
_output_shapes
:@*
dtype0
?
)Adam/module_wrapper_25/conv2d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)Adam/module_wrapper_25/conv2d_17/kernel/v
?
=Adam/module_wrapper_25/conv2d_17/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/module_wrapper_25/conv2d_17/kernel/v*&
_output_shapes
:@@*
dtype0
?
'Adam/module_wrapper_24/conv2d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'Adam/module_wrapper_24/conv2d_16/bias/v
?
;Adam/module_wrapper_24/conv2d_16/bias/v/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_24/conv2d_16/bias/v*
_output_shapes
:@*
dtype0
?
)Adam/module_wrapper_24/conv2d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*:
shared_name+)Adam/module_wrapper_24/conv2d_16/kernel/v
?
=Adam/module_wrapper_24/conv2d_16/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/module_wrapper_24/conv2d_16/kernel/v*&
_output_shapes
: @*
dtype0
?
'Adam/module_wrapper_22/conv2d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/module_wrapper_22/conv2d_15/bias/v
?
;Adam/module_wrapper_22/conv2d_15/bias/v/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_22/conv2d_15/bias/v*
_output_shapes
: *
dtype0
?
)Adam/module_wrapper_22/conv2d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *:
shared_name+)Adam/module_wrapper_22/conv2d_15/kernel/v
?
=Adam/module_wrapper_22/conv2d_15/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/module_wrapper_22/conv2d_15/kernel/v*&
_output_shapes
:  *
dtype0
?
'Adam/module_wrapper_21/conv2d_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/module_wrapper_21/conv2d_14/bias/v
?
;Adam/module_wrapper_21/conv2d_14/bias/v/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_21/conv2d_14/bias/v*
_output_shapes
: *
dtype0
?
)Adam/module_wrapper_21/conv2d_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)Adam/module_wrapper_21/conv2d_14/kernel/v
?
=Adam/module_wrapper_21/conv2d_14/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/module_wrapper_21/conv2d_14/kernel/v*&
_output_shapes
: *
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:
*
dtype0
?
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*&
shared_nameAdam/dense_9/kernel/v
?
)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes
:	?
*
dtype0

Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_8/bias/v
x
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_8/kernel/v
?
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v* 
_output_shapes
:
??*
dtype0
?
'Adam/module_wrapper_25/conv2d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'Adam/module_wrapper_25/conv2d_17/bias/m
?
;Adam/module_wrapper_25/conv2d_17/bias/m/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_25/conv2d_17/bias/m*
_output_shapes
:@*
dtype0
?
)Adam/module_wrapper_25/conv2d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)Adam/module_wrapper_25/conv2d_17/kernel/m
?
=Adam/module_wrapper_25/conv2d_17/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/module_wrapper_25/conv2d_17/kernel/m*&
_output_shapes
:@@*
dtype0
?
'Adam/module_wrapper_24/conv2d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'Adam/module_wrapper_24/conv2d_16/bias/m
?
;Adam/module_wrapper_24/conv2d_16/bias/m/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_24/conv2d_16/bias/m*
_output_shapes
:@*
dtype0
?
)Adam/module_wrapper_24/conv2d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*:
shared_name+)Adam/module_wrapper_24/conv2d_16/kernel/m
?
=Adam/module_wrapper_24/conv2d_16/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/module_wrapper_24/conv2d_16/kernel/m*&
_output_shapes
: @*
dtype0
?
'Adam/module_wrapper_22/conv2d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/module_wrapper_22/conv2d_15/bias/m
?
;Adam/module_wrapper_22/conv2d_15/bias/m/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_22/conv2d_15/bias/m*
_output_shapes
: *
dtype0
?
)Adam/module_wrapper_22/conv2d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *:
shared_name+)Adam/module_wrapper_22/conv2d_15/kernel/m
?
=Adam/module_wrapper_22/conv2d_15/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/module_wrapper_22/conv2d_15/kernel/m*&
_output_shapes
:  *
dtype0
?
'Adam/module_wrapper_21/conv2d_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/module_wrapper_21/conv2d_14/bias/m
?
;Adam/module_wrapper_21/conv2d_14/bias/m/Read/ReadVariableOpReadVariableOp'Adam/module_wrapper_21/conv2d_14/bias/m*
_output_shapes
: *
dtype0
?
)Adam/module_wrapper_21/conv2d_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)Adam/module_wrapper_21/conv2d_14/kernel/m
?
=Adam/module_wrapper_21/conv2d_14/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/module_wrapper_21/conv2d_14/kernel/m*&
_output_shapes
: *
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*&
shared_nameAdam/dense_9/kernel/m
?
)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes
:	?
*
dtype0

Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_8/bias/m
x
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_8/kernel/m
?
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m* 
_output_shapes
:
??*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
?
 module_wrapper_25/conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" module_wrapper_25/conv2d_17/bias
?
4module_wrapper_25/conv2d_17/bias/Read/ReadVariableOpReadVariableOp module_wrapper_25/conv2d_17/bias*
_output_shapes
:@*
dtype0
?
"module_wrapper_25/conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*3
shared_name$"module_wrapper_25/conv2d_17/kernel
?
6module_wrapper_25/conv2d_17/kernel/Read/ReadVariableOpReadVariableOp"module_wrapper_25/conv2d_17/kernel*&
_output_shapes
:@@*
dtype0
?
 module_wrapper_24/conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" module_wrapper_24/conv2d_16/bias
?
4module_wrapper_24/conv2d_16/bias/Read/ReadVariableOpReadVariableOp module_wrapper_24/conv2d_16/bias*
_output_shapes
:@*
dtype0
?
"module_wrapper_24/conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*3
shared_name$"module_wrapper_24/conv2d_16/kernel
?
6module_wrapper_24/conv2d_16/kernel/Read/ReadVariableOpReadVariableOp"module_wrapper_24/conv2d_16/kernel*&
_output_shapes
: @*
dtype0
?
 module_wrapper_22/conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" module_wrapper_22/conv2d_15/bias
?
4module_wrapper_22/conv2d_15/bias/Read/ReadVariableOpReadVariableOp module_wrapper_22/conv2d_15/bias*
_output_shapes
: *
dtype0
?
"module_wrapper_22/conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *3
shared_name$"module_wrapper_22/conv2d_15/kernel
?
6module_wrapper_22/conv2d_15/kernel/Read/ReadVariableOpReadVariableOp"module_wrapper_22/conv2d_15/kernel*&
_output_shapes
:  *
dtype0
?
 module_wrapper_21/conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" module_wrapper_21/conv2d_14/bias
?
4module_wrapper_21/conv2d_14/bias/Read/ReadVariableOpReadVariableOp module_wrapper_21/conv2d_14/bias*
_output_shapes
: *
dtype0
?
"module_wrapper_21/conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"module_wrapper_21/conv2d_14/kernel
?
6module_wrapper_21/conv2d_14/kernel/Read/ReadVariableOpReadVariableOp"module_wrapper_21/conv2d_14/kernel*&
_output_shapes
: *
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:
*
dtype0
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	?
*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:?*
dtype0
z
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
??*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_module*
?
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_module*
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*_module* 
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1_random_generator* 
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8_module*
?
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_module*
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_module* 
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
M_random_generator* 
?
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses* 
?
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias*
?
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
b_random_generator* 
?
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias*
Z
k0
l1
m2
n3
o4
p5
q6
r7
Z8
[9
i10
j11*
Z
k0
l1
m2
n3
o4
p5
q6
r7
Z8
[9
i10
j11*
* 
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
xtrace_0
ytrace_1
ztrace_2
{trace_3* 
6
|trace_0
}trace_1
~trace_2
trace_3* 
* 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateZm?[m?im?jm?km?lm?mm?nm?om?pm?qm?rm?Zv?[v?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?*

?serving_default* 

k0
l1*

k0
l1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kkernel
lbias*

m0
n1*

m0
n1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

mkernel
nbias*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

o0
p1*

o0
p1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

okernel
pbias*

q0
r1*

q0
r1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

qkernel
rbias*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

Z0
[1*

Z0
[1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 

i0
j1*

i0
j1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"module_wrapper_21/conv2d_14/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE module_wrapper_21/conv2d_14/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"module_wrapper_22/conv2d_15/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE module_wrapper_22/conv2d_15/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"module_wrapper_24/conv2d_16/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE module_wrapper_24/conv2d_16/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"module_wrapper_25/conv2d_17/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE module_wrapper_25/conv2d_17/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
Z
0
1
2
3
4
5
6
7
	8

9
10
11*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

k0
l1*
* 

k0
l1*
?
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

m0
n1*
* 

m0
n1*
?
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

o0
p1*
* 

o0
p1*
?
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

q0
r1*
* 

q0
r1*
?
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
?{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE)Adam/module_wrapper_21/conv2d_14/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUE'Adam/module_wrapper_21/conv2d_14/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE)Adam/module_wrapper_22/conv2d_15/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUE'Adam/module_wrapper_22/conv2d_15/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE)Adam/module_wrapper_24/conv2d_16/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUE'Adam/module_wrapper_24/conv2d_16/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE)Adam/module_wrapper_25/conv2d_17/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUE'Adam/module_wrapper_25/conv2d_17/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE)Adam/module_wrapper_21/conv2d_14/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUE'Adam/module_wrapper_21/conv2d_14/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE)Adam/module_wrapper_22/conv2d_15/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUE'Adam/module_wrapper_22/conv2d_15/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE)Adam/module_wrapper_24/conv2d_16/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUE'Adam/module_wrapper_24/conv2d_16/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
VARIABLE_VALUE)Adam/module_wrapper_25/conv2d_17/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUE'Adam/module_wrapper_25/conv2d_17/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
'serving_default_module_wrapper_21_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall'serving_default_module_wrapper_21_input"module_wrapper_21/conv2d_14/kernel module_wrapper_21/conv2d_14/bias"module_wrapper_22/conv2d_15/kernel module_wrapper_22/conv2d_15/bias"module_wrapper_24/conv2d_16/kernel module_wrapper_24/conv2d_16/bias"module_wrapper_25/conv2d_17/kernel module_wrapper_25/conv2d_17/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_314112
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp6module_wrapper_21/conv2d_14/kernel/Read/ReadVariableOp4module_wrapper_21/conv2d_14/bias/Read/ReadVariableOp6module_wrapper_22/conv2d_15/kernel/Read/ReadVariableOp4module_wrapper_22/conv2d_15/bias/Read/ReadVariableOp6module_wrapper_24/conv2d_16/kernel/Read/ReadVariableOp4module_wrapper_24/conv2d_16/bias/Read/ReadVariableOp6module_wrapper_25/conv2d_17/kernel/Read/ReadVariableOp4module_wrapper_25/conv2d_17/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp=Adam/module_wrapper_21/conv2d_14/kernel/m/Read/ReadVariableOp;Adam/module_wrapper_21/conv2d_14/bias/m/Read/ReadVariableOp=Adam/module_wrapper_22/conv2d_15/kernel/m/Read/ReadVariableOp;Adam/module_wrapper_22/conv2d_15/bias/m/Read/ReadVariableOp=Adam/module_wrapper_24/conv2d_16/kernel/m/Read/ReadVariableOp;Adam/module_wrapper_24/conv2d_16/bias/m/Read/ReadVariableOp=Adam/module_wrapper_25/conv2d_17/kernel/m/Read/ReadVariableOp;Adam/module_wrapper_25/conv2d_17/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp=Adam/module_wrapper_21/conv2d_14/kernel/v/Read/ReadVariableOp;Adam/module_wrapper_21/conv2d_14/bias/v/Read/ReadVariableOp=Adam/module_wrapper_22/conv2d_15/kernel/v/Read/ReadVariableOp;Adam/module_wrapper_22/conv2d_15/bias/v/Read/ReadVariableOp=Adam/module_wrapper_24/conv2d_16/kernel/v/Read/ReadVariableOp;Adam/module_wrapper_24/conv2d_16/bias/v/Read/ReadVariableOp=Adam/module_wrapper_25/conv2d_17/kernel/v/Read/ReadVariableOp;Adam/module_wrapper_25/conv2d_17/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_314811
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasdense_9/kerneldense_9/bias"module_wrapper_21/conv2d_14/kernel module_wrapper_21/conv2d_14/bias"module_wrapper_22/conv2d_15/kernel module_wrapper_22/conv2d_15/bias"module_wrapper_24/conv2d_16/kernel module_wrapper_24/conv2d_16/bias"module_wrapper_25/conv2d_17/kernel module_wrapper_25/conv2d_17/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/m)Adam/module_wrapper_21/conv2d_14/kernel/m'Adam/module_wrapper_21/conv2d_14/bias/m)Adam/module_wrapper_22/conv2d_15/kernel/m'Adam/module_wrapper_22/conv2d_15/bias/m)Adam/module_wrapper_24/conv2d_16/kernel/m'Adam/module_wrapper_24/conv2d_16/bias/m)Adam/module_wrapper_25/conv2d_17/kernel/m'Adam/module_wrapper_25/conv2d_17/bias/mAdam/dense_8/kernel/vAdam/dense_8/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/v)Adam/module_wrapper_21/conv2d_14/kernel/v'Adam/module_wrapper_21/conv2d_14/bias/v)Adam/module_wrapper_22/conv2d_15/kernel/v'Adam/module_wrapper_22/conv2d_15/bias/v)Adam/module_wrapper_24/conv2d_16/kernel/v'Adam/module_wrapper_24/conv2d_16/bias/v)Adam/module_wrapper_25/conv2d_17/kernel/v'Adam/module_wrapper_25/conv2d_17/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_314956??

?`
?
__inference__traced_save_314811
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableopA
=savev2_module_wrapper_21_conv2d_14_kernel_read_readvariableop?
;savev2_module_wrapper_21_conv2d_14_bias_read_readvariableopA
=savev2_module_wrapper_22_conv2d_15_kernel_read_readvariableop?
;savev2_module_wrapper_22_conv2d_15_bias_read_readvariableopA
=savev2_module_wrapper_24_conv2d_16_kernel_read_readvariableop?
;savev2_module_wrapper_24_conv2d_16_bias_read_readvariableopA
=savev2_module_wrapper_25_conv2d_17_kernel_read_readvariableop?
;savev2_module_wrapper_25_conv2d_17_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableopH
Dsavev2_adam_module_wrapper_21_conv2d_14_kernel_m_read_readvariableopF
Bsavev2_adam_module_wrapper_21_conv2d_14_bias_m_read_readvariableopH
Dsavev2_adam_module_wrapper_22_conv2d_15_kernel_m_read_readvariableopF
Bsavev2_adam_module_wrapper_22_conv2d_15_bias_m_read_readvariableopH
Dsavev2_adam_module_wrapper_24_conv2d_16_kernel_m_read_readvariableopF
Bsavev2_adam_module_wrapper_24_conv2d_16_bias_m_read_readvariableopH
Dsavev2_adam_module_wrapper_25_conv2d_17_kernel_m_read_readvariableopF
Bsavev2_adam_module_wrapper_25_conv2d_17_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableopH
Dsavev2_adam_module_wrapper_21_conv2d_14_kernel_v_read_readvariableopF
Bsavev2_adam_module_wrapper_21_conv2d_14_bias_v_read_readvariableopH
Dsavev2_adam_module_wrapper_22_conv2d_15_kernel_v_read_readvariableopF
Bsavev2_adam_module_wrapper_22_conv2d_15_bias_v_read_readvariableopH
Dsavev2_adam_module_wrapper_24_conv2d_16_kernel_v_read_readvariableopF
Bsavev2_adam_module_wrapper_24_conv2d_16_bias_v_read_readvariableopH
Dsavev2_adam_module_wrapper_25_conv2d_17_kernel_v_read_readvariableopF
Bsavev2_adam_module_wrapper_25_conv2d_17_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop=savev2_module_wrapper_21_conv2d_14_kernel_read_readvariableop;savev2_module_wrapper_21_conv2d_14_bias_read_readvariableop=savev2_module_wrapper_22_conv2d_15_kernel_read_readvariableop;savev2_module_wrapper_22_conv2d_15_bias_read_readvariableop=savev2_module_wrapper_24_conv2d_16_kernel_read_readvariableop;savev2_module_wrapper_24_conv2d_16_bias_read_readvariableop=savev2_module_wrapper_25_conv2d_17_kernel_read_readvariableop;savev2_module_wrapper_25_conv2d_17_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableopDsavev2_adam_module_wrapper_21_conv2d_14_kernel_m_read_readvariableopBsavev2_adam_module_wrapper_21_conv2d_14_bias_m_read_readvariableopDsavev2_adam_module_wrapper_22_conv2d_15_kernel_m_read_readvariableopBsavev2_adam_module_wrapper_22_conv2d_15_bias_m_read_readvariableopDsavev2_adam_module_wrapper_24_conv2d_16_kernel_m_read_readvariableopBsavev2_adam_module_wrapper_24_conv2d_16_bias_m_read_readvariableopDsavev2_adam_module_wrapper_25_conv2d_17_kernel_m_read_readvariableopBsavev2_adam_module_wrapper_25_conv2d_17_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableopDsavev2_adam_module_wrapper_21_conv2d_14_kernel_v_read_readvariableopBsavev2_adam_module_wrapper_21_conv2d_14_bias_v_read_readvariableopDsavev2_adam_module_wrapper_22_conv2d_15_kernel_v_read_readvariableopBsavev2_adam_module_wrapper_22_conv2d_15_bias_v_read_readvariableopDsavev2_adam_module_wrapper_24_conv2d_16_kernel_v_read_readvariableopBsavev2_adam_module_wrapper_24_conv2d_16_bias_v_read_readvariableopDsavev2_adam_module_wrapper_25_conv2d_17_kernel_v_read_readvariableopBsavev2_adam_module_wrapper_25_conv2d_17_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:	?
:
: : :  : : @:@:@@:@: : : : : : : : : :
??:?:	?
:
: : :  : : @:@:@@:@:
??:?:	?
:
: : :  : : @:@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,	(
&
_output_shapes
: @: 


_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:, (
&
_output_shapes
:@@: !

_output_shapes
:@:&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:%$!

_output_shapes
:	?
: %

_output_shapes
:
:,&(
&
_output_shapes
: : '

_output_shapes
: :,((
&
_output_shapes
:  : )

_output_shapes
: :,*(
&
_output_shapes
: @: +

_output_shapes
:@:,,(
&
_output_shapes
:@@: -

_output_shapes
:@:.

_output_shapes
: 
?
?
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_313831

args_0B
(conv2d_15_conv2d_readvariableop_resource:  7
)conv2d_15_biasadd_readvariableop_resource: 
identity?? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_15/Conv2DConv2Dargs_0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:????????? s
IdentityIdentityconv2d_15/Relu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?
?
2__inference_module_wrapper_22_layer_call_fn_314346

args_0!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_313476w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?4
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_313594

inputs2
module_wrapper_21_313460: &
module_wrapper_21_313462: 2
module_wrapper_22_313477:  &
module_wrapper_22_313479: 2
module_wrapper_24_313508: @&
module_wrapper_24_313510:@2
module_wrapper_25_313525:@@&
module_wrapper_25_313527:@"
dense_8_313564:
??
dense_8_313566:	?!
dense_9_313588:	?

dense_9_313590:

identity??dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?)module_wrapper_21/StatefulPartitionedCall?)module_wrapper_22/StatefulPartitionedCall?)module_wrapper_24/StatefulPartitionedCall?)module_wrapper_25/StatefulPartitionedCall?
)module_wrapper_21/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_21_313460module_wrapper_21_313462*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_313459?
)module_wrapper_22/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_21/StatefulPartitionedCall:output:0module_wrapper_22_313477module_wrapper_22_313479*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_313476?
!module_wrapper_23/PartitionedCallPartitionedCall2module_wrapper_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_313487?
dropout_11/PartitionedCallPartitionedCall*module_wrapper_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_313494?
)module_wrapper_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0module_wrapper_24_313508module_wrapper_24_313510*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_313507?
)module_wrapper_25/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_24/StatefulPartitionedCall:output:0module_wrapper_25_313525module_wrapper_25_313527*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_313524?
!module_wrapper_26/PartitionedCallPartitionedCall2module_wrapper_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_313535?
dropout_12/PartitionedCallPartitionedCall*module_wrapper_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_313542?
flatten_4/PartitionedCallPartitionedCall#dropout_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_313550?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_8_313564dense_8_313566*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_313563?
dropout_13/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_313574?
dense_9/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_9_313588dense_9_313590*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_313587w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*^module_wrapper_21/StatefulPartitionedCall*^module_wrapper_22/StatefulPartitionedCall*^module_wrapper_24/StatefulPartitionedCall*^module_wrapper_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2V
)module_wrapper_21/StatefulPartitionedCall)module_wrapper_21/StatefulPartitionedCall2V
)module_wrapper_22/StatefulPartitionedCall)module_wrapper_22/StatefulPartitionedCall2V
)module_wrapper_24/StatefulPartitionedCall)module_wrapper_24/StatefulPartitionedCall2V
)module_wrapper_25/StatefulPartitionedCall)module_wrapper_25/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

e
F__inference_dropout_11_layer_call_and_return_conditional_losses_314436

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
N
2__inference_module_wrapper_23_layer_call_fn_314382

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_313487h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_313524

args_0B
(conv2d_17_conv2d_readvariableop_resource:@@7
)conv2d_17_biasadd_readvariableop_resource:@
identity?? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_17/Conv2DConv2Dargs_0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_17/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_314465

args_0B
(conv2d_16_conv2d_readvariableop_resource: @7
)conv2d_16_biasadd_readvariableop_resource:@
identity?? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_16/Conv2DConv2Dargs_0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_16/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_313459

args_0B
(conv2d_14_conv2d_readvariableop_resource: 7
)conv2d_14_biasadd_readvariableop_resource: 
identity?? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_14/Conv2DConv2Dargs_0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? s
IdentityIdentityconv2d_14/Relu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
N
2__inference_module_wrapper_26_layer_call_fn_314526

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_313706h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
N
2__inference_module_wrapper_26_layer_call_fn_314521

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_313535h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
(__inference_dense_9_layer_call_fn_314642

inputs
unknown:	?

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_313587o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_314326

args_0B
(conv2d_14_conv2d_readvariableop_resource: 7
)conv2d_14_biasadd_readvariableop_resource: 
identity?? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_14/Conv2DConv2Dargs_0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? s
IdentityIdentityconv2d_14/Relu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
?
2__inference_module_wrapper_25_layer_call_fn_314494

args_0!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_313732w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
G
+__inference_dropout_13_layer_call_fn_314611

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_313574a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
2__inference_module_wrapper_24_layer_call_fn_314445

args_0!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_313507w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?
F
*__inference_flatten_4_layer_call_fn_314580

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_313550a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_313542

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?9
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_313939

inputs2
module_wrapper_21_313902: &
module_wrapper_21_313904: 2
module_wrapper_22_313907:  &
module_wrapper_22_313909: 2
module_wrapper_24_313914: @&
module_wrapper_24_313916:@2
module_wrapper_25_313919:@@&
module_wrapper_25_313921:@"
dense_8_313927:
??
dense_8_313929:	?!
dense_9_313933:	?

dense_9_313935:

identity??dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?"dropout_12/StatefulPartitionedCall?"dropout_13/StatefulPartitionedCall?)module_wrapper_21/StatefulPartitionedCall?)module_wrapper_22/StatefulPartitionedCall?)module_wrapper_24/StatefulPartitionedCall?)module_wrapper_25/StatefulPartitionedCall?
)module_wrapper_21/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_21_313902module_wrapper_21_313904*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_313861?
)module_wrapper_22/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_21/StatefulPartitionedCall:output:0module_wrapper_22_313907module_wrapper_22_313909*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_313831?
!module_wrapper_23/PartitionedCallPartitionedCall2module_wrapper_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_313805?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_313789?
)module_wrapper_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0module_wrapper_24_313914module_wrapper_24_313916*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_313762?
)module_wrapper_25/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_24/StatefulPartitionedCall:output:0module_wrapper_25_313919module_wrapper_25_313921*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_313732?
!module_wrapper_26/PartitionedCallPartitionedCall2module_wrapper_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_313706?
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_26/PartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_313690?
flatten_4/PartitionedCallPartitionedCall+dropout_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_313550?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_8_313927dense_8_313929*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_313563?
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_313651?
dense_9/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_9_313933dense_9_313935*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_313587w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall*^module_wrapper_21/StatefulPartitionedCall*^module_wrapper_22/StatefulPartitionedCall*^module_wrapper_24/StatefulPartitionedCall*^module_wrapper_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2V
)module_wrapper_21/StatefulPartitionedCall)module_wrapper_21/StatefulPartitionedCall2V
)module_wrapper_22/StatefulPartitionedCall)module_wrapper_22/StatefulPartitionedCall2V
)module_wrapper_24/StatefulPartitionedCall)module_wrapper_24/StatefulPartitionedCall2V
)module_wrapper_25/StatefulPartitionedCall)module_wrapper_25/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
i
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_314397

args_0
identity?
max_pooling2d_7/MaxPoolMaxPoolargs_0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
p
IdentityIdentity max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_313762

args_0B
(conv2d_16_conv2d_readvariableop_resource: @7
)conv2d_16_biasadd_readvariableop_resource:@
identity?? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_16/Conv2DConv2Dargs_0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_16/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?9
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_314075
module_wrapper_21_input2
module_wrapper_21_314038: &
module_wrapper_21_314040: 2
module_wrapper_22_314043:  &
module_wrapper_22_314045: 2
module_wrapper_24_314050: @&
module_wrapper_24_314052:@2
module_wrapper_25_314055:@@&
module_wrapper_25_314057:@"
dense_8_314063:
??
dense_8_314065:	?!
dense_9_314069:	?

dense_9_314071:

identity??dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?"dropout_11/StatefulPartitionedCall?"dropout_12/StatefulPartitionedCall?"dropout_13/StatefulPartitionedCall?)module_wrapper_21/StatefulPartitionedCall?)module_wrapper_22/StatefulPartitionedCall?)module_wrapper_24/StatefulPartitionedCall?)module_wrapper_25/StatefulPartitionedCall?
)module_wrapper_21/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_21_inputmodule_wrapper_21_314038module_wrapper_21_314040*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_313861?
)module_wrapper_22/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_21/StatefulPartitionedCall:output:0module_wrapper_22_314043module_wrapper_22_314045*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_313831?
!module_wrapper_23/PartitionedCallPartitionedCall2module_wrapper_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_313805?
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_313789?
)module_wrapper_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0module_wrapper_24_314050module_wrapper_24_314052*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_313762?
)module_wrapper_25/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_24/StatefulPartitionedCall:output:0module_wrapper_25_314055module_wrapper_25_314057*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_313732?
!module_wrapper_26/PartitionedCallPartitionedCall2module_wrapper_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_313706?
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall*module_wrapper_26/PartitionedCall:output:0#^dropout_11/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_313690?
flatten_4/PartitionedCallPartitionedCall+dropout_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_313550?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_8_314063dense_8_314065*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_313563?
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_313651?
dense_9/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_9_314069dense_9_314071*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_313587w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall*^module_wrapper_21/StatefulPartitionedCall*^module_wrapper_22/StatefulPartitionedCall*^module_wrapper_24/StatefulPartitionedCall*^module_wrapper_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2V
)module_wrapper_21/StatefulPartitionedCall)module_wrapper_21/StatefulPartitionedCall2V
)module_wrapper_22/StatefulPartitionedCall)module_wrapper_22/StatefulPartitionedCall2V
)module_wrapper_24/StatefulPartitionedCall)module_wrapper_24/StatefulPartitionedCall2V
)module_wrapper_25/StatefulPartitionedCall)module_wrapper_25/StatefulPartitionedCall:h d
/
_output_shapes
:?????????
1
_user_specified_namemodule_wrapper_21_input
?
i
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_313535

args_0
identity?
max_pooling2d_8/MaxPoolMaxPoolargs_0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
p
IdentityIdentity max_pooling2d_8/MaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
2__inference_module_wrapper_25_layer_call_fn_314485

args_0!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_313524w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
-__inference_sequential_4_layer_call_fn_314170

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_313939o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_module_wrapper_21_layer_call_fn_314306

args_0!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_313459w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
N
2__inference_module_wrapper_23_layer_call_fn_314387

args_0
identity?
PartitionedCallPartitionedCallargs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_313805h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?

?
C__inference_dense_8_layer_call_and_return_conditional_losses_314606

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
e
F__inference_dropout_13_layer_call_and_return_conditional_losses_313651

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_12_layer_call_fn_314553

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_313542h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

e
F__inference_dropout_12_layer_call_and_return_conditional_losses_314575

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
i
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_313706

args_0
identity?
max_pooling2d_8/MaxPoolMaxPoolargs_0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
p
IdentityIdentity max_pooling2d_8/MaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_314377

args_0B
(conv2d_15_conv2d_readvariableop_resource:  7
)conv2d_15_biasadd_readvariableop_resource: 
identity?? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_15/Conv2DConv2Dargs_0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:????????? s
IdentityIdentityconv2d_15/Relu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_313476

args_0B
(conv2d_15_conv2d_readvariableop_resource:  7
)conv2d_15_biasadd_readvariableop_resource: 
identity?? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_15/Conv2DConv2Dargs_0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:????????? s
IdentityIdentityconv2d_15/Relu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_314621

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_8_layer_call_fn_314595

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_313563p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
e
F__inference_dropout_13_layer_call_and_return_conditional_losses_314633

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_314424

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
2__inference_module_wrapper_22_layer_call_fn_314355

args_0!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_313831w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?

?
C__inference_dense_8_layer_call_and_return_conditional_losses_313563

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
+__inference_dropout_12_layer_call_fn_314558

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_313690w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
i
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_313805

args_0
identity?
max_pooling2d_7/MaxPoolMaxPoolargs_0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
p
IdentityIdentity max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?
d
F__inference_dropout_11_layer_call_and_return_conditional_losses_313494

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
-__inference_sequential_4_layer_call_fn_313621
module_wrapper_21_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_313594o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
/
_output_shapes
:?????????
1
_user_specified_namemodule_wrapper_21_input
?
?
$__inference_signature_wrapper_314112
module_wrapper_21_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_313441o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
/
_output_shapes
:?????????
1
_user_specified_namemodule_wrapper_21_input
?

e
F__inference_dropout_12_layer_call_and_return_conditional_losses_313690

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
-__inference_sequential_4_layer_call_fn_314141

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_313594o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?L
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_314223

inputsT
:module_wrapper_21_conv2d_14_conv2d_readvariableop_resource: I
;module_wrapper_21_conv2d_14_biasadd_readvariableop_resource: T
:module_wrapper_22_conv2d_15_conv2d_readvariableop_resource:  I
;module_wrapper_22_conv2d_15_biasadd_readvariableop_resource: T
:module_wrapper_24_conv2d_16_conv2d_readvariableop_resource: @I
;module_wrapper_24_conv2d_16_biasadd_readvariableop_resource:@T
:module_wrapper_25_conv2d_17_conv2d_readvariableop_resource:@@I
;module_wrapper_25_conv2d_17_biasadd_readvariableop_resource:@:
&dense_8_matmul_readvariableop_resource:
??6
'dense_8_biasadd_readvariableop_resource:	?9
&dense_9_matmul_readvariableop_resource:	?
5
'dense_9_biasadd_readvariableop_resource:

identity??dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?2module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp?1module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp?2module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp?1module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp?2module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp?1module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp?2module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp?1module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp?
1module_wrapper_21/conv2d_14/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_21_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
"module_wrapper_21/conv2d_14/Conv2DConv2Dinputs9module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
2module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_21_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
#module_wrapper_21/conv2d_14/BiasAddBiasAdd+module_wrapper_21/conv2d_14/Conv2D:output:0:module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
 module_wrapper_21/conv2d_14/ReluRelu,module_wrapper_21/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
1module_wrapper_22/conv2d_15/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_22_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
"module_wrapper_22/conv2d_15/Conv2DConv2D.module_wrapper_21/conv2d_14/Relu:activations:09module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
2module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_22_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
#module_wrapper_22/conv2d_15/BiasAddBiasAdd+module_wrapper_22/conv2d_15/Conv2D:output:0:module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
 module_wrapper_22/conv2d_15/ReluRelu,module_wrapper_22/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
)module_wrapper_23/max_pooling2d_7/MaxPoolMaxPool.module_wrapper_22/conv2d_15/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
dropout_11/IdentityIdentity2module_wrapper_23/max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:????????? ?
1module_wrapper_24/conv2d_16/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_24_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
"module_wrapper_24/conv2d_16/Conv2DConv2Ddropout_11/Identity:output:09module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
2module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_24_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
#module_wrapper_24/conv2d_16/BiasAddBiasAdd+module_wrapper_24/conv2d_16/Conv2D:output:0:module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
 module_wrapper_24/conv2d_16/ReluRelu,module_wrapper_24/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
1module_wrapper_25/conv2d_17/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_25_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
"module_wrapper_25/conv2d_17/Conv2DConv2D.module_wrapper_24/conv2d_16/Relu:activations:09module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
2module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_25_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
#module_wrapper_25/conv2d_17/BiasAddBiasAdd+module_wrapper_25/conv2d_17/Conv2D:output:0:module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
 module_wrapper_25/conv2d_17/ReluRelu,module_wrapper_25/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
)module_wrapper_26/max_pooling2d_8/MaxPoolMaxPool.module_wrapper_25/conv2d_17/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
dropout_12/IdentityIdentity2module_wrapper_26/max_pooling2d_8/MaxPool:output:0*
T0*/
_output_shapes
:?????????@`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
flatten_4/ReshapeReshapedropout_12/Identity:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:???????????
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_8/MatMulMatMulflatten_4/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????n
dropout_13/IdentityIdentitydense_8/Relu:activations:0*
T0*(
_output_shapes
:???????????
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense_9/MatMulMatMuldropout_13/Identity:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
f
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
h
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp3^module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp2^module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp3^module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp2^module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp3^module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp2^module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp3^module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp2^module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2h
2module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp2module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp2f
1module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp1module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp2h
2module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp2module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp2f
1module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp1module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp2h
2module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp2module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp2f
1module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp1module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp2h
2module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp2module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp2f
1module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp1module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_module_wrapper_24_layer_call_fn_314454

args_0!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_313762w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?
L
0__inference_max_pooling2d_7_layer_call_fn_314409

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_314403?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_314586

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

e
F__inference_dropout_11_layer_call_and_return_conditional_losses_313789

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
2__inference_module_wrapper_21_layer_call_fn_314315

args_0!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_313861w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_314505

args_0B
(conv2d_17_conv2d_readvariableop_resource:@@7
)conv2d_17_biasadd_readvariableop_resource:@
identity?? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_17/Conv2DConv2Dargs_0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_17/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
g
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_314403

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_314366

args_0B
(conv2d_15_conv2d_readvariableop_resource:  7
)conv2d_15_biasadd_readvariableop_resource: 
identity?? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_15/Conv2DConv2Dargs_0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:????????? s
IdentityIdentityconv2d_15/Relu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_314337

args_0B
(conv2d_14_conv2d_readvariableop_resource: 7
)conv2d_14_biasadd_readvariableop_resource: 
identity?? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_14/Conv2DConv2Dargs_0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? s
IdentityIdentityconv2d_14/Relu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_314476

args_0B
(conv2d_16_conv2d_readvariableop_resource: @7
)conv2d_16_biasadd_readvariableop_resource:@
identity?? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_16/Conv2DConv2Dargs_0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_16/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?

?
C__inference_dense_9_layer_call_and_return_conditional_losses_314653

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?[
?
!__inference__wrapped_model_313441
module_wrapper_21_inputa
Gsequential_4_module_wrapper_21_conv2d_14_conv2d_readvariableop_resource: V
Hsequential_4_module_wrapper_21_conv2d_14_biasadd_readvariableop_resource: a
Gsequential_4_module_wrapper_22_conv2d_15_conv2d_readvariableop_resource:  V
Hsequential_4_module_wrapper_22_conv2d_15_biasadd_readvariableop_resource: a
Gsequential_4_module_wrapper_24_conv2d_16_conv2d_readvariableop_resource: @V
Hsequential_4_module_wrapper_24_conv2d_16_biasadd_readvariableop_resource:@a
Gsequential_4_module_wrapper_25_conv2d_17_conv2d_readvariableop_resource:@@V
Hsequential_4_module_wrapper_25_conv2d_17_biasadd_readvariableop_resource:@G
3sequential_4_dense_8_matmul_readvariableop_resource:
??C
4sequential_4_dense_8_biasadd_readvariableop_resource:	?F
3sequential_4_dense_9_matmul_readvariableop_resource:	?
B
4sequential_4_dense_9_biasadd_readvariableop_resource:

identity??+sequential_4/dense_8/BiasAdd/ReadVariableOp?*sequential_4/dense_8/MatMul/ReadVariableOp?+sequential_4/dense_9/BiasAdd/ReadVariableOp?*sequential_4/dense_9/MatMul/ReadVariableOp??sequential_4/module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp?>sequential_4/module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp??sequential_4/module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp?>sequential_4/module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp??sequential_4/module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp?>sequential_4/module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp??sequential_4/module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp?>sequential_4/module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp?
>sequential_4/module_wrapper_21/conv2d_14/Conv2D/ReadVariableOpReadVariableOpGsequential_4_module_wrapper_21_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
/sequential_4/module_wrapper_21/conv2d_14/Conv2DConv2Dmodule_wrapper_21_inputFsequential_4/module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
?sequential_4/module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOpReadVariableOpHsequential_4_module_wrapper_21_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
0sequential_4/module_wrapper_21/conv2d_14/BiasAddBiasAdd8sequential_4/module_wrapper_21/conv2d_14/Conv2D:output:0Gsequential_4/module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
-sequential_4/module_wrapper_21/conv2d_14/ReluRelu9sequential_4/module_wrapper_21/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
>sequential_4/module_wrapper_22/conv2d_15/Conv2D/ReadVariableOpReadVariableOpGsequential_4_module_wrapper_22_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
/sequential_4/module_wrapper_22/conv2d_15/Conv2DConv2D;sequential_4/module_wrapper_21/conv2d_14/Relu:activations:0Fsequential_4/module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
?sequential_4/module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOpReadVariableOpHsequential_4_module_wrapper_22_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
0sequential_4/module_wrapper_22/conv2d_15/BiasAddBiasAdd8sequential_4/module_wrapper_22/conv2d_15/Conv2D:output:0Gsequential_4/module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
-sequential_4/module_wrapper_22/conv2d_15/ReluRelu9sequential_4/module_wrapper_22/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
6sequential_4/module_wrapper_23/max_pooling2d_7/MaxPoolMaxPool;sequential_4/module_wrapper_22/conv2d_15/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
?
 sequential_4/dropout_11/IdentityIdentity?sequential_4/module_wrapper_23/max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:????????? ?
>sequential_4/module_wrapper_24/conv2d_16/Conv2D/ReadVariableOpReadVariableOpGsequential_4_module_wrapper_24_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
/sequential_4/module_wrapper_24/conv2d_16/Conv2DConv2D)sequential_4/dropout_11/Identity:output:0Fsequential_4/module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
?sequential_4/module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOpReadVariableOpHsequential_4_module_wrapper_24_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
0sequential_4/module_wrapper_24/conv2d_16/BiasAddBiasAdd8sequential_4/module_wrapper_24/conv2d_16/Conv2D:output:0Gsequential_4/module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
-sequential_4/module_wrapper_24/conv2d_16/ReluRelu9sequential_4/module_wrapper_24/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
>sequential_4/module_wrapper_25/conv2d_17/Conv2D/ReadVariableOpReadVariableOpGsequential_4_module_wrapper_25_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
/sequential_4/module_wrapper_25/conv2d_17/Conv2DConv2D;sequential_4/module_wrapper_24/conv2d_16/Relu:activations:0Fsequential_4/module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
?sequential_4/module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOpReadVariableOpHsequential_4_module_wrapper_25_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
0sequential_4/module_wrapper_25/conv2d_17/BiasAddBiasAdd8sequential_4/module_wrapper_25/conv2d_17/Conv2D:output:0Gsequential_4/module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
-sequential_4/module_wrapper_25/conv2d_17/ReluRelu9sequential_4/module_wrapper_25/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
6sequential_4/module_wrapper_26/max_pooling2d_8/MaxPoolMaxPool;sequential_4/module_wrapper_25/conv2d_17/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
 sequential_4/dropout_12/IdentityIdentity?sequential_4/module_wrapper_26/max_pooling2d_8/MaxPool:output:0*
T0*/
_output_shapes
:?????????@m
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_12/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:???????????
*sequential_4/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_4/dense_8/MatMulMatMul'sequential_4/flatten_4/Reshape:output:02sequential_4/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
+sequential_4/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_4/dense_8/BiasAddBiasAdd%sequential_4/dense_8/MatMul:product:03sequential_4/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????{
sequential_4/dense_8/ReluRelu%sequential_4/dense_8/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
 sequential_4/dropout_13/IdentityIdentity'sequential_4/dense_8/Relu:activations:0*
T0*(
_output_shapes
:???????????
*sequential_4/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_9_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
sequential_4/dense_9/MatMulMatMul)sequential_4/dropout_13/Identity:output:02sequential_4/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
+sequential_4/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
sequential_4/dense_9/BiasAddBiasAdd%sequential_4/dense_9/MatMul:product:03sequential_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
sequential_4/dense_9/SoftmaxSoftmax%sequential_4/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
u
IdentityIdentity&sequential_4/dense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp,^sequential_4/dense_8/BiasAdd/ReadVariableOp+^sequential_4/dense_8/MatMul/ReadVariableOp,^sequential_4/dense_9/BiasAdd/ReadVariableOp+^sequential_4/dense_9/MatMul/ReadVariableOp@^sequential_4/module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp?^sequential_4/module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp@^sequential_4/module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp?^sequential_4/module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp@^sequential_4/module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp?^sequential_4/module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp@^sequential_4/module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp?^sequential_4/module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2Z
+sequential_4/dense_8/BiasAdd/ReadVariableOp+sequential_4/dense_8/BiasAdd/ReadVariableOp2X
*sequential_4/dense_8/MatMul/ReadVariableOp*sequential_4/dense_8/MatMul/ReadVariableOp2Z
+sequential_4/dense_9/BiasAdd/ReadVariableOp+sequential_4/dense_9/BiasAdd/ReadVariableOp2X
*sequential_4/dense_9/MatMul/ReadVariableOp*sequential_4/dense_9/MatMul/ReadVariableOp2?
?sequential_4/module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp?sequential_4/module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp2?
>sequential_4/module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp>sequential_4/module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp2?
?sequential_4/module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp?sequential_4/module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp2?
>sequential_4/module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp>sequential_4/module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp2?
?sequential_4/module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp?sequential_4/module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp2?
>sequential_4/module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp>sequential_4/module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp2?
?sequential_4/module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp?sequential_4/module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp2?
>sequential_4/module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp>sequential_4/module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp:h d
/
_output_shapes
:?????????
1
_user_specified_namemodule_wrapper_21_input
?

?
C__inference_dense_9_layer_call_and_return_conditional_losses_313587

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?5
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_314035
module_wrapper_21_input2
module_wrapper_21_313998: &
module_wrapper_21_314000: 2
module_wrapper_22_314003:  &
module_wrapper_22_314005: 2
module_wrapper_24_314010: @&
module_wrapper_24_314012:@2
module_wrapper_25_314015:@@&
module_wrapper_25_314017:@"
dense_8_314023:
??
dense_8_314025:	?!
dense_9_314029:	?

dense_9_314031:

identity??dense_8/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?)module_wrapper_21/StatefulPartitionedCall?)module_wrapper_22/StatefulPartitionedCall?)module_wrapper_24/StatefulPartitionedCall?)module_wrapper_25/StatefulPartitionedCall?
)module_wrapper_21/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_21_inputmodule_wrapper_21_313998module_wrapper_21_314000*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_313459?
)module_wrapper_22/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_21/StatefulPartitionedCall:output:0module_wrapper_22_314003module_wrapper_22_314005*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_313476?
!module_wrapper_23/PartitionedCallPartitionedCall2module_wrapper_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_313487?
dropout_11/PartitionedCallPartitionedCall*module_wrapper_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_313494?
)module_wrapper_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0module_wrapper_24_314010module_wrapper_24_314012*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_313507?
)module_wrapper_25/StatefulPartitionedCallStatefulPartitionedCall2module_wrapper_24/StatefulPartitionedCall:output:0module_wrapper_25_314015module_wrapper_25_314017*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_313524?
!module_wrapper_26/PartitionedCallPartitionedCall2module_wrapper_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_313535?
dropout_12/PartitionedCallPartitionedCall*module_wrapper_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_313542?
flatten_4/PartitionedCallPartitionedCall#dropout_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_4_layer_call_and_return_conditional_losses_313550?
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_8_314023dense_8_314025*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_313563?
dropout_13/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_313574?
dense_9/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_9_314029dense_9_314031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_313587w
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*^module_wrapper_21/StatefulPartitionedCall*^module_wrapper_22/StatefulPartitionedCall*^module_wrapper_24/StatefulPartitionedCall*^module_wrapper_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2V
)module_wrapper_21/StatefulPartitionedCall)module_wrapper_21/StatefulPartitionedCall2V
)module_wrapper_22/StatefulPartitionedCall)module_wrapper_22/StatefulPartitionedCall2V
)module_wrapper_24/StatefulPartitionedCall)module_wrapper_24/StatefulPartitionedCall2V
)module_wrapper_25/StatefulPartitionedCall)module_wrapper_25/StatefulPartitionedCall:h d
/
_output_shapes
:?????????
1
_user_specified_namemodule_wrapper_21_input
?
?
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_313507

args_0B
(conv2d_16_conv2d_readvariableop_resource: @7
)conv2d_16_biasadd_readvariableop_resource:@
identity?? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_16/Conv2DConv2Dargs_0'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_16/ReluReluconv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_16/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_314392

args_0
identity?
max_pooling2d_7/MaxPoolMaxPoolargs_0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
p
IdentityIdentity max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
??
?
"__inference__traced_restore_314956
file_prefix3
assignvariableop_dense_8_kernel:
??.
assignvariableop_1_dense_8_bias:	?4
!assignvariableop_2_dense_9_kernel:	?
-
assignvariableop_3_dense_9_bias:
O
5assignvariableop_4_module_wrapper_21_conv2d_14_kernel: A
3assignvariableop_5_module_wrapper_21_conv2d_14_bias: O
5assignvariableop_6_module_wrapper_22_conv2d_15_kernel:  A
3assignvariableop_7_module_wrapper_22_conv2d_15_bias: O
5assignvariableop_8_module_wrapper_24_conv2d_16_kernel: @A
3assignvariableop_9_module_wrapper_24_conv2d_16_bias:@P
6assignvariableop_10_module_wrapper_25_conv2d_17_kernel:@@B
4assignvariableop_11_module_wrapper_25_conv2d_17_bias:@'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: #
assignvariableop_19_total: #
assignvariableop_20_count: =
)assignvariableop_21_adam_dense_8_kernel_m:
??6
'assignvariableop_22_adam_dense_8_bias_m:	?<
)assignvariableop_23_adam_dense_9_kernel_m:	?
5
'assignvariableop_24_adam_dense_9_bias_m:
W
=assignvariableop_25_adam_module_wrapper_21_conv2d_14_kernel_m: I
;assignvariableop_26_adam_module_wrapper_21_conv2d_14_bias_m: W
=assignvariableop_27_adam_module_wrapper_22_conv2d_15_kernel_m:  I
;assignvariableop_28_adam_module_wrapper_22_conv2d_15_bias_m: W
=assignvariableop_29_adam_module_wrapper_24_conv2d_16_kernel_m: @I
;assignvariableop_30_adam_module_wrapper_24_conv2d_16_bias_m:@W
=assignvariableop_31_adam_module_wrapper_25_conv2d_17_kernel_m:@@I
;assignvariableop_32_adam_module_wrapper_25_conv2d_17_bias_m:@=
)assignvariableop_33_adam_dense_8_kernel_v:
??6
'assignvariableop_34_adam_dense_8_bias_v:	?<
)assignvariableop_35_adam_dense_9_kernel_v:	?
5
'assignvariableop_36_adam_dense_9_bias_v:
W
=assignvariableop_37_adam_module_wrapper_21_conv2d_14_kernel_v: I
;assignvariableop_38_adam_module_wrapper_21_conv2d_14_bias_v: W
=assignvariableop_39_adam_module_wrapper_22_conv2d_15_kernel_v:  I
;assignvariableop_40_adam_module_wrapper_22_conv2d_15_bias_v: W
=assignvariableop_41_adam_module_wrapper_24_conv2d_16_kernel_v: @I
;assignvariableop_42_adam_module_wrapper_24_conv2d_16_bias_v:@W
=assignvariableop_43_adam_module_wrapper_25_conv2d_17_kernel_v:@@I
;assignvariableop_44_adam_module_wrapper_25_conv2d_17_bias_v:@
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_module_wrapper_21_conv2d_14_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp3assignvariableop_5_module_wrapper_21_conv2d_14_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp5assignvariableop_6_module_wrapper_22_conv2d_15_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp3assignvariableop_7_module_wrapper_22_conv2d_15_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp5assignvariableop_8_module_wrapper_24_conv2d_16_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp3assignvariableop_9_module_wrapper_24_conv2d_16_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_module_wrapper_25_conv2d_17_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp4assignvariableop_11_module_wrapper_25_conv2d_17_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_8_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_8_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_9_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_9_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp=assignvariableop_25_adam_module_wrapper_21_conv2d_14_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp;assignvariableop_26_adam_module_wrapper_21_conv2d_14_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp=assignvariableop_27_adam_module_wrapper_22_conv2d_15_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp;assignvariableop_28_adam_module_wrapper_22_conv2d_15_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp=assignvariableop_29_adam_module_wrapper_24_conv2d_16_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp;assignvariableop_30_adam_module_wrapper_24_conv2d_16_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp=assignvariableop_31_adam_module_wrapper_25_conv2d_17_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp;assignvariableop_32_adam_module_wrapper_25_conv2d_17_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_8_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_8_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_9_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_9_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp=assignvariableop_37_adam_module_wrapper_21_conv2d_14_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp;assignvariableop_38_adam_module_wrapper_21_conv2d_14_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp=assignvariableop_39_adam_module_wrapper_22_conv2d_15_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp;assignvariableop_40_adam_module_wrapper_22_conv2d_15_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp=assignvariableop_41_adam_module_wrapper_24_conv2d_16_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp;assignvariableop_42_adam_module_wrapper_24_conv2d_16_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp=assignvariableop_43_adam_module_wrapper_25_conv2d_17_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp;assignvariableop_44_adam_module_wrapper_25_conv2d_17_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
a
E__inference_flatten_4_layer_call_and_return_conditional_losses_313550

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
-__inference_sequential_4_layer_call_fn_313995
module_wrapper_21_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
??
	unknown_8:	?
	unknown_9:	?


unknown_10:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_21_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_313939o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
/
_output_shapes
:?????????
1
_user_specified_namemodule_wrapper_21_input
?
d
+__inference_dropout_11_layer_call_fn_314419

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_313789w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?d
?
H__inference_sequential_4_layer_call_and_return_conditional_losses_314297

inputsT
:module_wrapper_21_conv2d_14_conv2d_readvariableop_resource: I
;module_wrapper_21_conv2d_14_biasadd_readvariableop_resource: T
:module_wrapper_22_conv2d_15_conv2d_readvariableop_resource:  I
;module_wrapper_22_conv2d_15_biasadd_readvariableop_resource: T
:module_wrapper_24_conv2d_16_conv2d_readvariableop_resource: @I
;module_wrapper_24_conv2d_16_biasadd_readvariableop_resource:@T
:module_wrapper_25_conv2d_17_conv2d_readvariableop_resource:@@I
;module_wrapper_25_conv2d_17_biasadd_readvariableop_resource:@:
&dense_8_matmul_readvariableop_resource:
??6
'dense_8_biasadd_readvariableop_resource:	?9
&dense_9_matmul_readvariableop_resource:	?
5
'dense_9_biasadd_readvariableop_resource:

identity??dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?2module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp?1module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp?2module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp?1module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp?2module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp?1module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp?2module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp?1module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp?
1module_wrapper_21/conv2d_14/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_21_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
"module_wrapper_21/conv2d_14/Conv2DConv2Dinputs9module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
2module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_21_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
#module_wrapper_21/conv2d_14/BiasAddBiasAdd+module_wrapper_21/conv2d_14/Conv2D:output:0:module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
 module_wrapper_21/conv2d_14/ReluRelu,module_wrapper_21/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
1module_wrapper_22/conv2d_15/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_22_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
"module_wrapper_22/conv2d_15/Conv2DConv2D.module_wrapper_21/conv2d_14/Relu:activations:09module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
2module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_22_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
#module_wrapper_22/conv2d_15/BiasAddBiasAdd+module_wrapper_22/conv2d_15/Conv2D:output:0:module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
 module_wrapper_22/conv2d_15/ReluRelu,module_wrapper_22/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
)module_wrapper_23/max_pooling2d_7/MaxPoolMaxPool.module_wrapper_22/conv2d_15/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
]
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
dropout_11/dropout/MulMul2module_wrapper_23/max_pooling2d_7/MaxPool:output:0!dropout_11/dropout/Const:output:0*
T0*/
_output_shapes
:????????? z
dropout_11/dropout/ShapeShape2module_wrapper_23/max_pooling2d_7/MaxPool:output:0*
T0*
_output_shapes
:?
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0f
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? ?
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? ?
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? ?
1module_wrapper_24/conv2d_16/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_24_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
"module_wrapper_24/conv2d_16/Conv2DConv2Ddropout_11/dropout/Mul_1:z:09module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
2module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_24_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
#module_wrapper_24/conv2d_16/BiasAddBiasAdd+module_wrapper_24/conv2d_16/Conv2D:output:0:module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
 module_wrapper_24/conv2d_16/ReluRelu,module_wrapper_24/conv2d_16/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
1module_wrapper_25/conv2d_17/Conv2D/ReadVariableOpReadVariableOp:module_wrapper_25_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
"module_wrapper_25/conv2d_17/Conv2DConv2D.module_wrapper_24/conv2d_16/Relu:activations:09module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
2module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp;module_wrapper_25_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
#module_wrapper_25/conv2d_17/BiasAddBiasAdd+module_wrapper_25/conv2d_17/Conv2D:output:0:module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?
 module_wrapper_25/conv2d_17/ReluRelu,module_wrapper_25/conv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
)module_wrapper_26/max_pooling2d_8/MaxPoolMaxPool.module_wrapper_25/conv2d_17/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
]
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
dropout_12/dropout/MulMul2module_wrapper_26/max_pooling2d_8/MaxPool:output:0!dropout_12/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@z
dropout_12/dropout/ShapeShape2module_wrapper_26/max_pooling2d_8/MaxPool:output:0*
T0*
_output_shapes
:?
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype0f
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@?
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@?
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
flatten_4/ReshapeReshapedropout_12/dropout/Mul_1:z:0flatten_4/Const:output:0*
T0*(
_output_shapes
:???????????
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_8/MatMulMatMulflatten_4/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*(
_output_shapes
:??????????]
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout_13/dropout/MulMuldense_8/Relu:activations:0!dropout_13/dropout/Const:output:0*
T0*(
_output_shapes
:??????????b
dropout_13/dropout/ShapeShapedense_8/Relu:activations:0*
T0*
_output_shapes
:?
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0f
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense_9/MatMulMatMuldropout_13/dropout/Mul_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
f
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
h
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp3^module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp2^module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp3^module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp2^module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp3^module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp2^module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp3^module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp2^module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:?????????: : : : : : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2h
2module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp2module_wrapper_21/conv2d_14/BiasAdd/ReadVariableOp2f
1module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp1module_wrapper_21/conv2d_14/Conv2D/ReadVariableOp2h
2module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp2module_wrapper_22/conv2d_15/BiasAdd/ReadVariableOp2f
1module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp1module_wrapper_22/conv2d_15/Conv2D/ReadVariableOp2h
2module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp2module_wrapper_24/conv2d_16/BiasAdd/ReadVariableOp2f
1module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp1module_wrapper_24/conv2d_16/Conv2D/ReadVariableOp2h
2module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp2module_wrapper_25/conv2d_17/BiasAdd/ReadVariableOp2f
1module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp1module_wrapper_25/conv2d_17/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_314563

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
i
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_313487

args_0
identity?
max_pooling2d_7/MaxPoolMaxPoolargs_0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
p
IdentityIdentity max_pooling2d_7/MaxPool:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameargs_0
?
d
F__inference_dropout_13_layer_call_and_return_conditional_losses_313574

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_11_layer_call_fn_314414

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_11_layer_call_and_return_conditional_losses_313494h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
d
+__inference_dropout_13_layer_call_fn_314616

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_13_layer_call_and_return_conditional_losses_313651p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_314516

args_0B
(conv2d_17_conv2d_readvariableop_resource:@@7
)conv2d_17_biasadd_readvariableop_resource:@
identity?? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_17/Conv2DConv2Dargs_0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_17/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
?
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_313861

args_0B
(conv2d_14_conv2d_readvariableop_resource: 7
)conv2d_14_biasadd_readvariableop_resource: 
identity?? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_14/Conv2DConv2Dargs_0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? s
IdentityIdentityconv2d_14/Relu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_314536

args_0
identity?
max_pooling2d_8/MaxPoolMaxPoolargs_0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
p
IdentityIdentity max_pooling2d_8/MaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
L
0__inference_max_pooling2d_8_layer_call_fn_314548

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_314542?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_313732

args_0B
(conv2d_17_conv2d_readvariableop_resource:@@7
)conv2d_17_biasadd_readvariableop_resource:@
identity?? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_17/Conv2DConv2Dargs_0'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_17/ReluReluconv2d_17/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@s
IdentityIdentityconv2d_17/Relu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
i
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_314531

args_0
identity?
max_pooling2d_8/MaxPoolMaxPoolargs_0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
p
IdentityIdentity max_pooling2d_8/MaxPool:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameargs_0
?
g
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_314542

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
c
module_wrapper_21_inputH
)serving_default_module_wrapper_21_input:0?????????;
dense_90
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_module"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_module"
_tf_keras_layer
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*_module"
_tf_keras_layer
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses
1_random_generator"
_tf_keras_layer
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8_module"
_tf_keras_layer
?
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_module"
_tf_keras_layer
?
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
F_module"
_tf_keras_layer
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
M_random_generator"
_tf_keras_layer
?
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
?
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias"
_tf_keras_layer
?
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
b_random_generator"
_tf_keras_layer
?
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias"
_tf_keras_layer
v
k0
l1
m2
n3
o4
p5
q6
r7
Z8
[9
i10
j11"
trackable_list_wrapper
v
k0
l1
m2
n3
o4
p5
q6
r7
Z8
[9
i10
j11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
xtrace_0
ytrace_1
ztrace_2
{trace_32?
-__inference_sequential_4_layer_call_fn_313621
-__inference_sequential_4_layer_call_fn_314141
-__inference_sequential_4_layer_call_fn_314170
-__inference_sequential_4_layer_call_fn_313995?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zxtrace_0zytrace_1zztrace_2z{trace_3
?
|trace_0
}trace_1
~trace_2
trace_32?
H__inference_sequential_4_layer_call_and_return_conditional_losses_314223
H__inference_sequential_4_layer_call_and_return_conditional_losses_314297
H__inference_sequential_4_layer_call_and_return_conditional_losses_314035
H__inference_sequential_4_layer_call_and_return_conditional_losses_314075?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z|trace_0z}trace_1z~trace_2ztrace_3
?B?
!__inference__wrapped_model_313441module_wrapper_21_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateZm?[m?im?jm?km?lm?mm?nm?om?pm?qm?rm?Zv?[v?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?"
	optimizer
-
?serving_default"
signature_map
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_21_layer_call_fn_314306
2__inference_module_wrapper_21_layer_call_fn_314315?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_314326
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_314337?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

kkernel
lbias"
_tf_keras_layer
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_22_layer_call_fn_314346
2__inference_module_wrapper_22_layer_call_fn_314355?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_314366
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_314377?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

mkernel
nbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_23_layer_call_fn_314382
2__inference_module_wrapper_23_layer_call_fn_314387?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_314392
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_314397?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
+__inference_dropout_11_layer_call_fn_314414
+__inference_dropout_11_layer_call_fn_314419?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
F__inference_dropout_11_layer_call_and_return_conditional_losses_314424
F__inference_dropout_11_layer_call_and_return_conditional_losses_314436?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_24_layer_call_fn_314445
2__inference_module_wrapper_24_layer_call_fn_314454?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_314465
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_314476?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

okernel
pbias"
_tf_keras_layer
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_25_layer_call_fn_314485
2__inference_module_wrapper_25_layer_call_fn_314494?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_314505
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_314516?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses

qkernel
rbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
2__inference_module_wrapper_26_layer_call_fn_314521
2__inference_module_wrapper_26_layer_call_fn_314526?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_314531
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_314536?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z?trace_0z?trace_1
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
+__inference_dropout_12_layer_call_fn_314553
+__inference_dropout_12_layer_call_fn_314558?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
F__inference_dropout_12_layer_call_and_return_conditional_losses_314563
F__inference_dropout_12_layer_call_and_return_conditional_losses_314575?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_flatten_4_layer_call_fn_314580?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_flatten_4_layer_call_and_return_conditional_losses_314586?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_dense_8_layer_call_fn_314595?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_dense_8_layer_call_and_return_conditional_losses_314606?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
": 
??2dense_8/kernel
:?2dense_8/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
+__inference_dropout_13_layer_call_fn_314611
+__inference_dropout_13_layer_call_fn_314616?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
F__inference_dropout_13_layer_call_and_return_conditional_losses_314621
F__inference_dropout_13_layer_call_and_return_conditional_losses_314633?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_dense_9_layer_call_fn_314642?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_dense_9_layer_call_and_return_conditional_losses_314653?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:	?
2dense_9/kernel
:
2dense_9/bias
<:: 2"module_wrapper_21/conv2d_14/kernel
.:, 2 module_wrapper_21/conv2d_14/bias
<::  2"module_wrapper_22/conv2d_15/kernel
.:, 2 module_wrapper_22/conv2d_15/bias
<:: @2"module_wrapper_24/conv2d_16/kernel
.:,@2 module_wrapper_24/conv2d_16/bias
<::@@2"module_wrapper_25/conv2d_17/kernel
.:,@2 module_wrapper_25/conv2d_17/bias
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
-__inference_sequential_4_layer_call_fn_313621module_wrapper_21_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_sequential_4_layer_call_fn_314141inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_sequential_4_layer_call_fn_314170inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
-__inference_sequential_4_layer_call_fn_313995module_wrapper_21_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_sequential_4_layer_call_and_return_conditional_losses_314223inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_sequential_4_layer_call_and_return_conditional_losses_314297inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_sequential_4_layer_call_and_return_conditional_losses_314035module_wrapper_21_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
H__inference_sequential_4_layer_call_and_return_conditional_losses_314075module_wrapper_21_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
$__inference_signature_wrapper_314112module_wrapper_21_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
2__inference_module_wrapper_21_layer_call_fn_314306args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_21_layer_call_fn_314315args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_314326args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_314337args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
?
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
2__inference_module_wrapper_22_layer_call_fn_314346args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_22_layer_call_fn_314355args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_314366args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_314377args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
?
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
2__inference_module_wrapper_23_layer_call_fn_314382args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_23_layer_call_fn_314387args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_314392args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_314397args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
0__inference_max_pooling2d_7_layer_call_fn_314409?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????z?trace_0
?
?trace_02?
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_314403?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_dropout_11_layer_call_fn_314414inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
+__inference_dropout_11_layer_call_fn_314419inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_dropout_11_layer_call_and_return_conditional_losses_314424inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_dropout_11_layer_call_and_return_conditional_losses_314436inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
2__inference_module_wrapper_24_layer_call_fn_314445args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_24_layer_call_fn_314454args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_314465args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_314476args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
?
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
2__inference_module_wrapper_25_layer_call_fn_314485args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_25_layer_call_fn_314494args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_314505args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_314516args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
?
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
2__inference_module_wrapper_26_layer_call_fn_314521args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
2__inference_module_wrapper_26_layer_call_fn_314526args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_314531args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_314536args_0"?
???
FullArgSpec
args?
jself
varargsjargs
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
 ?layer_regularization_losses
?layer_metrics
?metrics
?regularization_losses
?	variables
?non_trainable_variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
0__inference_max_pooling2d_8_layer_call_fn_314548?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????z?trace_0
?
?trace_02?
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_314542?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_dropout_12_layer_call_fn_314553inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
+__inference_dropout_12_layer_call_fn_314558inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_dropout_12_layer_call_and_return_conditional_losses_314563inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_dropout_12_layer_call_and_return_conditional_losses_314575inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_flatten_4_layer_call_fn_314580inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_flatten_4_layer_call_and_return_conditional_losses_314586inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_dense_8_layer_call_fn_314595inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_dense_8_layer_call_and_return_conditional_losses_314606inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_dropout_13_layer_call_fn_314611inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
+__inference_dropout_13_layer_call_fn_314616inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_dropout_13_layer_call_and_return_conditional_losses_314621inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
F__inference_dropout_13_layer_call_and_return_conditional_losses_314633inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_dense_9_layer_call_fn_314642inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_dense_9_layer_call_and_return_conditional_losses_314653inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?B?
0__inference_max_pooling2d_7_layer_call_fn_314409inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?B?
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_314403inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?B?
0__inference_max_pooling2d_8_layer_call_fn_314548inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?B?
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_314542inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
':%
??2Adam/dense_8/kernel/m
 :?2Adam/dense_8/bias/m
&:$	?
2Adam/dense_9/kernel/m
:
2Adam/dense_9/bias/m
A:? 2)Adam/module_wrapper_21/conv2d_14/kernel/m
3:1 2'Adam/module_wrapper_21/conv2d_14/bias/m
A:?  2)Adam/module_wrapper_22/conv2d_15/kernel/m
3:1 2'Adam/module_wrapper_22/conv2d_15/bias/m
A:? @2)Adam/module_wrapper_24/conv2d_16/kernel/m
3:1@2'Adam/module_wrapper_24/conv2d_16/bias/m
A:?@@2)Adam/module_wrapper_25/conv2d_17/kernel/m
3:1@2'Adam/module_wrapper_25/conv2d_17/bias/m
':%
??2Adam/dense_8/kernel/v
 :?2Adam/dense_8/bias/v
&:$	?
2Adam/dense_9/kernel/v
:
2Adam/dense_9/bias/v
A:? 2)Adam/module_wrapper_21/conv2d_14/kernel/v
3:1 2'Adam/module_wrapper_21/conv2d_14/bias/v
A:?  2)Adam/module_wrapper_22/conv2d_15/kernel/v
3:1 2'Adam/module_wrapper_22/conv2d_15/bias/v
A:? @2)Adam/module_wrapper_24/conv2d_16/kernel/v
3:1@2'Adam/module_wrapper_24/conv2d_16/bias/v
A:?@@2)Adam/module_wrapper_25/conv2d_17/kernel/v
3:1@2'Adam/module_wrapper_25/conv2d_17/bias/v?
!__inference__wrapped_model_313441?klmnopqrZ[ijH?E
>?;
9?6
module_wrapper_21_input?????????
? "1?.
,
dense_9!?
dense_9?????????
?
C__inference_dense_8_layer_call_and_return_conditional_losses_314606^Z[0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? }
(__inference_dense_8_layer_call_fn_314595QZ[0?-
&?#
!?
inputs??????????
? "????????????
C__inference_dense_9_layer_call_and_return_conditional_losses_314653]ij0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? |
(__inference_dense_9_layer_call_fn_314642Pij0?-
&?#
!?
inputs??????????
? "??????????
?
F__inference_dropout_11_layer_call_and_return_conditional_losses_314424l;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
F__inference_dropout_11_layer_call_and_return_conditional_losses_314436l;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
+__inference_dropout_11_layer_call_fn_314414_;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
+__inference_dropout_11_layer_call_fn_314419_;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
F__inference_dropout_12_layer_call_and_return_conditional_losses_314563l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
F__inference_dropout_12_layer_call_and_return_conditional_losses_314575l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
+__inference_dropout_12_layer_call_fn_314553_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
+__inference_dropout_12_layer_call_fn_314558_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
F__inference_dropout_13_layer_call_and_return_conditional_losses_314621^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
F__inference_dropout_13_layer_call_and_return_conditional_losses_314633^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
+__inference_dropout_13_layer_call_fn_314611Q4?1
*?'
!?
inputs??????????
p 
? "????????????
+__inference_dropout_13_layer_call_fn_314616Q4?1
*?'
!?
inputs??????????
p
? "????????????
E__inference_flatten_4_layer_call_and_return_conditional_losses_314586a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
*__inference_flatten_4_layer_call_fn_314580T7?4
-?*
(?%
inputs?????????@
? "????????????
K__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_314403?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_7_layer_call_fn_314409?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_314542?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_8_layer_call_fn_314548?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_314326|klG?D
-?*
(?%
args_0?????????
?

trainingp "-?*
#? 
0????????? 
? ?
M__inference_module_wrapper_21_layer_call_and_return_conditional_losses_314337|klG?D
-?*
(?%
args_0?????????
?

trainingp"-?*
#? 
0????????? 
? ?
2__inference_module_wrapper_21_layer_call_fn_314306oklG?D
-?*
(?%
args_0?????????
?

trainingp " ?????????? ?
2__inference_module_wrapper_21_layer_call_fn_314315oklG?D
-?*
(?%
args_0?????????
?

trainingp" ?????????? ?
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_314366|mnG?D
-?*
(?%
args_0????????? 
?

trainingp "-?*
#? 
0????????? 
? ?
M__inference_module_wrapper_22_layer_call_and_return_conditional_losses_314377|mnG?D
-?*
(?%
args_0????????? 
?

trainingp"-?*
#? 
0????????? 
? ?
2__inference_module_wrapper_22_layer_call_fn_314346omnG?D
-?*
(?%
args_0????????? 
?

trainingp " ?????????? ?
2__inference_module_wrapper_22_layer_call_fn_314355omnG?D
-?*
(?%
args_0????????? 
?

trainingp" ?????????? ?
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_314392xG?D
-?*
(?%
args_0????????? 
?

trainingp "-?*
#? 
0????????? 
? ?
M__inference_module_wrapper_23_layer_call_and_return_conditional_losses_314397xG?D
-?*
(?%
args_0????????? 
?

trainingp"-?*
#? 
0????????? 
? ?
2__inference_module_wrapper_23_layer_call_fn_314382kG?D
-?*
(?%
args_0????????? 
?

trainingp " ?????????? ?
2__inference_module_wrapper_23_layer_call_fn_314387kG?D
-?*
(?%
args_0????????? 
?

trainingp" ?????????? ?
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_314465|opG?D
-?*
(?%
args_0????????? 
?

trainingp "-?*
#? 
0?????????@
? ?
M__inference_module_wrapper_24_layer_call_and_return_conditional_losses_314476|opG?D
-?*
(?%
args_0????????? 
?

trainingp"-?*
#? 
0?????????@
? ?
2__inference_module_wrapper_24_layer_call_fn_314445oopG?D
-?*
(?%
args_0????????? 
?

trainingp " ??????????@?
2__inference_module_wrapper_24_layer_call_fn_314454oopG?D
-?*
(?%
args_0????????? 
?

trainingp" ??????????@?
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_314505|qrG?D
-?*
(?%
args_0?????????@
?

trainingp "-?*
#? 
0?????????@
? ?
M__inference_module_wrapper_25_layer_call_and_return_conditional_losses_314516|qrG?D
-?*
(?%
args_0?????????@
?

trainingp"-?*
#? 
0?????????@
? ?
2__inference_module_wrapper_25_layer_call_fn_314485oqrG?D
-?*
(?%
args_0?????????@
?

trainingp " ??????????@?
2__inference_module_wrapper_25_layer_call_fn_314494oqrG?D
-?*
(?%
args_0?????????@
?

trainingp" ??????????@?
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_314531xG?D
-?*
(?%
args_0?????????@
?

trainingp "-?*
#? 
0?????????@
? ?
M__inference_module_wrapper_26_layer_call_and_return_conditional_losses_314536xG?D
-?*
(?%
args_0?????????@
?

trainingp"-?*
#? 
0?????????@
? ?
2__inference_module_wrapper_26_layer_call_fn_314521kG?D
-?*
(?%
args_0?????????@
?

trainingp " ??????????@?
2__inference_module_wrapper_26_layer_call_fn_314526kG?D
-?*
(?%
args_0?????????@
?

trainingp" ??????????@?
H__inference_sequential_4_layer_call_and_return_conditional_losses_314035?klmnopqrZ[ijP?M
F?C
9?6
module_wrapper_21_input?????????
p 

 
? "%?"
?
0?????????

? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_314075?klmnopqrZ[ijP?M
F?C
9?6
module_wrapper_21_input?????????
p

 
? "%?"
?
0?????????

? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_314223vklmnopqrZ[ij??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
H__inference_sequential_4_layer_call_and_return_conditional_losses_314297vklmnopqrZ[ij??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????

? ?
-__inference_sequential_4_layer_call_fn_313621zklmnopqrZ[ijP?M
F?C
9?6
module_wrapper_21_input?????????
p 

 
? "??????????
?
-__inference_sequential_4_layer_call_fn_313995zklmnopqrZ[ijP?M
F?C
9?6
module_wrapper_21_input?????????
p

 
? "??????????
?
-__inference_sequential_4_layer_call_fn_314141iklmnopqrZ[ij??<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
-__inference_sequential_4_layer_call_fn_314170iklmnopqrZ[ij??<
5?2
(?%
inputs?????????
p

 
? "??????????
?
$__inference_signature_wrapper_314112?klmnopqrZ[ijc?`
? 
Y?V
T
module_wrapper_21_input9?6
module_wrapper_21_input?????????"1?.
,
dense_9!?
dense_9?????????
