import json, argparse, time

import tensorflow as tf
import numpy as np
from load import load_graph

from flask import Flask, request

##################################################
# API part
##################################################
app = Flask(__name__)
@app.route("/api/predict", methods=['POST'])
def predict():
    start = time.time()

    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        x_in = json.loads(params['x'])
    else:
        params = json.loads(data)
        x_in = params['x']

    x_in  = np.reshape(x_in, (28, 28, 1))
    ##################################################
    # Tensorflow part
    ##################################################
    y_out = persistent_sess.run(y, feed_dict={
        x: [x_in]
    })
    averaged_y_out = y_out.mean(0)
    argmax = np.argmax(averaged_y_out)
    max = np.amax(averaged_y_out)
    output = np.concatenate((np.array([argmax], float), np.array([max], float)))
    ##################################################
    # END Tensorflow part
    ##################################################
    json_data = json.dumps({'y': output.tolist()})
    print("Time spent handling the request: %f" % (time.time() - start))

    return json_data
##################################################
# END API part
##################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="mnist_model/frozen_model.pb", type=str, help="Frozen model file to import")
    parser.add_argument("--gpu_memory", default=.2, type=float, help="GPU memory per process")
    args = parser.parse_args()

    ##################################################
    # Tensorflow part
    ##################################################
    print('Loading the model')
    graph = load_graph(args.frozen_model_filename)
    print(graph.get_operations())
    x = graph.get_tensor_by_name('prefix/Reshape:0')
    y = graph.get_tensor_by_name('prefix/softmax_tensor:0')
    print('Starting Session, setting the GPU memory usage to %f' % args.gpu_memory)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    persistent_sess = tf.Session(graph=graph, config=sess_config)
    ##################################################
    # END Tensorflow part
    ##################################################

    print('Starting the API')
    app.run()
