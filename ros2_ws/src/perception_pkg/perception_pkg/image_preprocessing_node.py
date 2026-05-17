import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ImagePreprocessingNode(Node):
    def __init__(self):
        super().__init__('image_preprocessing_node')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            '/pylon_camera_node/image_raw',
            self.cb,
            10
        )
        self.pub = self.create_publisher(
            Image,
            '/pylon_camera_node/image_color',
            10
        )

    def cb(self, msg:Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg,desired_encoding='passthrough')

        # OPTION 2:
        color = cv2.cvtColor(cv_img, cv2.COLOR_BayerBG2BGR)

        out = self.bridge.cv2_to_imgmsg(color, encoding='bgr8')
        out.header = msg.header
        self.pub.publish(out)

def main():
    rclpy.init()
    node = ImagePreprocessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()