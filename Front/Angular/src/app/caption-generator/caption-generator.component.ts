import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-caption-generator',
  templateUrl: './caption-generator.component.html',
  styleUrls: ['./caption-generator.component.css']
})
export class CaptionGeneratorComponent implements OnInit {

  file!: File;

  constructor(private http: HttpClient) { }

  ngOnInit(): void {
  }


  OnImageChange(event: any){
    this.file= event.target.files[0];
    console.log(this.file)
  }
  OnCaptionSubmit(){
    const uploadData =new FormData();
    uploadData.append('file',this.file);

    this.http.post('http://35.202.9.1:5000/api/upload/captionimage',uploadData).subscribe(
      data => console.log(data),
      error => console.log(error)
    );
  }

}
