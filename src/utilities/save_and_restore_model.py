#Save trained model
save_path = saver.save(sess, "/content/ckp/model2.ckpt")
print("Model saved in path: %s" % save_path)

#Restore the trained model and just DONT declare the variable again!
saver.restore(sess, "/content/ckp/model2.ckpt")
print("Model restored.") 
