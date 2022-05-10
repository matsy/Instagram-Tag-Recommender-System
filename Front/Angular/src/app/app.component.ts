import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';



@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit{
  
  title = 'Insta-Tag';
  // Inject service
	constructor(public router: Router) { 
  }
  ngOnInit(): void {
  }
  public onButton1(){
  
    this.router.navigate(['/app-tag-generator']);
  }
  public onButton2(){
   
    this.router.navigate(['/app-caption-generator']);
  }
  public onButton3(){
    this.router.navigate(['/app-hashtag-list']);
  }

   

// OnClick of button Upload

}

  

