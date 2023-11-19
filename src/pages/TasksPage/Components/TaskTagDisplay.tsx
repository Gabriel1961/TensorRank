import { Component } from "solid-js"
import styles from "./TaskTagDisplay.module.scss"
import {TaskTag} from "../../../datamodel"
interface TaskTagDisplayProps{
  tag:TaskTag
}

const TaskTagDisplay: Component<TaskTagDisplayProps> = (props) => {
  const getTagColor = () => {
    switch (props.tag){
      case TaskTag.classification: return "#34495e"
      case TaskTag.images: return "#1d5c85"
      case TaskTag.regression: return "#8e44ad"
      case TaskTag.multivariate: return "#1d5c85";
    } 
  }
  return <div class={styles.container} style={`background-color:${getTagColor()};`} >
    {props.tag}
  </div>
}

export default TaskTagDisplay;