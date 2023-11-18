import { Component } from "solid-js";
import styles from "./AboutPage.module.scss"

const AboutPage: Component = () => {
  return <div class={styles.container}>
    <h2>Scope</h2>
    <p>
      This platform is a good resource for learning and practicing your Tensorflow skills. It is anologus to Hackerank, but it's going to be for tensorflow. Here you can learn simple and complex algorithms like regression and classification using tensorflow. 
    </p>
  </div>
}

export default AboutPage;